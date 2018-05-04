var app = angular.module('DashboardApp', ['ngRoute', 'ngMaterial']);

app.config(function ($locationProvider) {
  $locationProvider.html5Mode(true);
})

app.filter('padding', function () {
  return function(num, width) {
    return ("0000" + num).slice(-width);
  };
});

app.filter('integer', function () {
  return function(num) {
    if (!isFinite(num)) return "---";
    return Math.round(num).toString();
  };
});

app.controller('DashboardCtrl', function($scope, $location, $rootScope, $interval,
                                         $mdToast) {
  $scope.auto = false;
  $scope.freq = 15;
  $scope.done = false;
  $scope.error = false;
  var ticks = 0;

  $scope.host = function() {
    return $location.host() + ":" + $location.port();
  };

  $rootScope.refresh = function() {
    $scope.error = false;
    $rootScope.$emit('refresh');
  };

  $scope.play = function(state, rate) {
    $scope.auto = state;
    $scope.error = false;
    if (rate != undefined) $scope.freq = rate;
    ticks = 0;
  }

  $interval(function() {
    ticks++;
    if ($scope.auto && ticks++ % $scope.freq == 0) {
      $scope.error = false;
      $rootScope.$emit('refresh');
    }
  }, 1000);

  $rootScope.$on('error', function(event, args) {
    if (!$scope.error) {
      $mdToast.show($mdToast.simple()
          .textContent('Connection to server lost')
          .hideDelay(3000)
      );
    }
    $scope.error = true;
    $scope.auto = false;
  });

  $rootScope.$on('done', function(event, args) {
    if (!$scope.done) {
      $mdToast.show($mdToast.simple()
          .textContent('Workflow completed')
          .hideDelay(3000)
      );
    }
    $scope.error = false;
    $scope.done = true;
    $scope.auto = false;
  });
})

app.controller('StatusCtrl', function($scope, $http, $rootScope) {
  $scope.running = true;
  $scope.status = null;
  $scope.jobs = [];
  $scope.selected = 0;
  $scope.census = null;

  var resources = null;

  function updateTitle(msg) {
    var host = window.location.hostname + ":" + window.location.port;
    window.document.title = msg + " - SLING jobs on " + host;
  }

  $rootScope.$on('refresh', function(event, args) {
    $scope.refresh();
  });

  $scope.update = function() {
    var status = $scope.status

    // Update resource usage.
    var runtime = status.time - status.started;
    var res = {};
    res.time = status.time;
    res.cpu = (status.utime + status.stime) / 1000000;
    res.gflops = status.flops / 1e9;
    res.ram = status.mem;
    res.io = status.ioread + status.iowrite;
    var census = {};
    census.hours = Math.floor(runtime / 3600);
    census.mins = Math.floor((runtime % 3600) / 60);
    census.secs = Math.floor(runtime % 60);
    census.temp = status.temperature;
    if (status.finished) {
      census.cpu = res.cpu / runtime;
      census.gflops = res.gflops / runtime;
      census.ram = res.ram;
      census.io = res.io / runtime;
    } else if (resources) {
      var dt = res.time - resources.time;
      census.cpu = (res.cpu - resources.cpu) / dt;
      census.gflops = (res.gflops - resources.gflops) / dt;
      census.ram = res.ram;
      census.io = (res.io - resources.io) / dt;
    }
    $scope.census = census;
    resources = res;

    // Update job list.
    for (var i = 0; i < status.jobs.length; ++i) {
      var jobstatus = status.jobs[i];

      // Create new job tab if needed.
      var job = $scope.jobs[i];
      if (job == undefined) {
        job = {};
        job.id = i;
        job.name = jobstatus.name;
        job.prev_counters = null;
        job.prev_channels = null;
        job.prev_time = null;
        $scope.jobs[i] = job;
        $scope.selected = i;
        updateTitle(job.name);
      }

      // Compute elapsed time for job.
      var ended = jobstatus.ended ? jobstatus.ended : status.time;
      var elapsed = ended - jobstatus.started;
      job.hours = Math.floor(elapsed / 3600);
      job.mins = Math.floor((elapsed % 3600) / 60);
      job.secs = Math.floor(elapsed % 60);

      // Compute task progress for job.
      var progress = "";
      if (jobstatus.stages) {
        for (var j = 0; j < jobstatus.stages.length; ++j) {
          var stage = jobstatus.stages[j];
          if (j > 0) progress += "│ ";
          progress += "█ ".repeat(stage.done);
          progress += "░ ".repeat(stage.tasks - stage.done);
        }
      } else {
        progress = "✔";
      }
      job.progress = progress;

      // Process job counters.
      var counters = [];
      var channels = [];
      var channel_map = {};
      var prev_counters = job.prev_counters;
      var prev_channels = job.prev_channels;
      var period = status.time - job.prev_time;
      var channel_pattern = /(input|output)_(.+)\[(.+\..+)\]/;
      for (name in jobstatus.counters) {
        // Check for channel stat counter.
        m = name.match(channel_pattern);
        if (m) {
          // Look up channel.
          var direction = m[1];
          var metric = m[2];
          var channel_name = m[3];
          var ch = channel_map[channel_name];
          if (ch == null) {
            ch = {};
            ch.name = channel_name;
            ch.direction = direction;
            channel_map[channel_name] = ch;
            channels.push(ch);
          }

          // Update channel metrics.
          var value = jobstatus.counters[name];
          if (metric == "key_bytes") {
            ch.key_bytes = value;
          } else if (metric == "value_bytes") {
            ch.value_bytes = value;
          } else if (metric == "messages") {
            ch.messages = value;
          } else if (metric == "shards") {
            ch.shards_total = value;
          } else if (metric == "shards_done") {
            ch.shards_done = value;
          }
        } else {
          // Add counter.
          item = {};
          item.name = name;
          item.value = jobstatus.counters[name];
          if (jobstatus.ended) {
            item.rate = item.value / elapsed;
          } else if (prev_counters && period > 0) {
            var prev_value = prev_counters[name];
            var delta = item.value - prev_value;
            item.rate = delta / period;
          }
          counters.push(item);
        }
      }

      // Compute bandwidth and throughput.
      if (prev_channels) {
        for (var j = 0; j < channels.length; ++j) {
          var ch = channels[j];
          var prev = prev_channels[ch.name];
          if (jobstatus.ended) {
            ch.bandwidth = (ch.key_bytes + ch.value_bytes) / elapsed;
            ch.throughput = ch.messages / elapsed;
          } else if (prev) {
            var current_bytes = ch.key_bytes + ch.value_bytes;
            var prev_bytes = prev.key_bytes + prev.value_bytes;
            ch.bandwidth = (current_bytes - prev_bytes) / period;
            ch.throughput = (ch.messages - prev.messages) / period;
          }
        }
      }

      // Update job.
      job.counters = counters;
      job.channels = channels;
      job.prev_counters = jobstatus.counters;
      job.prev_channels = channel_map;
      job.prev_time = status.time;

      // Check for workflow completed.
      if (status.finished) {
        $scope.running = false;
        $rootScope.$emit('done');
        updateTitle("Done");
      }
    }
  }

  $scope.refresh = function() {
    $http.get('/status').then(function(response) {
      $scope.status = response.data;
      $scope.update();
    }, function(response) {
      $rootScope.$emit('error');
      updateTitle("Error");
    });
  }

  $scope.refresh();
})

