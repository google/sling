var app = angular.module('WikicatBrowserApp', ['ngRoute', 'ngMaterial']);

app.config(function ($locationProvider) {
  $locationProvider.html5Mode(true);
})

// Top-level controller.
app.controller('SearchCtrl', function($scope, $http,  $location, $mdDialog) {
  $scope.match_types = [];       // mirrors FactMatchType names
  $scope.default_weights = {};   // default weights for match types
  $scope.user_weights = {};      // user-specified weights for match types

  // First, get the list of match types and their default weights.
  $http.get("/wikicat/weights")
    .then(function(response) {
      for (var i = 0; i < response.data.length; ++i) {
        var t = response.data[i][0];
        var wt = response.data[i][1];
        $scope.match_types.push(t);
        $scope.default_weights[t] = wt;
        $scope.user_weights[t] = wt;
      }
  });

  // Get basic counts.
  $http.get("/wikicat/basic")
    .then(function(response) {
      $scope.num_parses = response.data["num_parses"];
      $scope.num_categories = response.data["num_categories"];
  });

  $scope.query = "";                // main text box
  $scope.query_results = null;      // results of the query
  $scope.recordio_results = null;   // result of recordio generation
  $scope.parse_selector = "";       // user-specified parse-selection formula

  // Function that shows help-text for writing parse selection formulae.
  $scope.formulaHelp = function(ev) {
    ev.stopPropagation();
    $mdDialog.show({
      contentElement: '#formulaHelpDialog',
      parent: angular.element(document.body),
      targetEvent: ev,
      clickOutsideToClose: true
    });
  }

  // Returns whether there was an error in processing the previous query.
  $scope.hasError = function() {
    return $scope.query_results != null &&
      typeof($scope.query_results.error) !== 'undefined';
  }

  // Returns true if the previous query searched for a signature.
  $scope.hasSignatureResults = function() {
    return $scope.query_results != null &&
      $scope.query_results.response_type == "signature" &&
      typeof($scope.query_results.error) == 'undefined';
  }

  // Returns true if the previous operation dealt with recordio generation.
  $scope.hasRecordioResults = function() {
    return $scope.recordio_results != null;
  }

  // Returns true if the previous query searched for a category.
  $scope.hasCategoryResults = function() {
    return $scope.query_results != null &&
      $scope.query_results.response_type == "category";
  }

  // Returns true if the previous query searched for top-signatures.
  $scope.hasTopResults = function() {
    return $scope.query_results != null &&
      $scope.query_results.response_type == "top";
  }

  // Constructs and returns the URL for querying the server.
  $scope.makeUrlParams = function() {
    if ($scope.query == "") return null;

    // Collect the list of spans (only relevant for signature search).
    var span_subset = [];
    if ($scope.hasSignatureResults()) {
      spans = $scope.query_results.per_span;
      for (var i = 0; i < spans.length; ++i) {
        if (spans[i].selected) span_subset.push(i);
      }
    }
    var span_subset_str = span_subset.join(',');
    if (span_subset_str == "") span_subset_str = "-1";  // all spans selected

    // Collect the list of match-type weights.
    var metric = "";
    for (var match_type in $scope.user_weights) {
      if (metric != "") metric += ",";
      metric += match_type + ":" + $scope.user_weights[match_type];
    }

    // Parse selector formula.
    var selector = "True";
    if ($scope.parse_selector != "") selector = $scope.parse_selector;

    var params = {
      "query": $scope.query,
      "spans": span_subset_str,
      "metric": metric,
      "selector": selector
    };
    var params_list = []
    for (var param in params) {
      params_list.push(encodeURIComponent(param) + "=" +
        encodeURIComponent(params[param]));
    }
    return params_list.join("&");
  }

  // Main workhorse. Issues a query to the server and fetches the results.
  $scope.search = function() {
    url_params = $scope.makeUrlParams();
    if (url_params == null) return;

    // Show a busy cursor while we wait for the response.
    document.body.classList.add('waiting');

    return $http.get('/wikicat/query?' + url_params)
      .then(function(results) {
        $scope.query_results = results.data;
        $scope.recordio_results = null;
        document.body.classList.remove('waiting');
      });
  }

  // Issues a recordio generation request to the server.
  $scope.recordio = function() {
    if (!$scope.hasSignatureResults()) return;

    url_params = $scope.makeUrlParams();
    if (url_params == null) return;

    return $http.get('/wikicat/recordio?' + url_params)
      .then(function(results) {
        $scope.recordio_results = results.data;
      });
  }

  // Click-handler for navigating to a signature. Shows all parses for the
  // clicked signature by clearing the parse-selection formula.
  $scope.browseSignature = function(sig) {
    $scope.query = sig;
    $scope.parse_selector = "True";
    $scope.search();
  }

  // Click-handler for navigating to a category. Shows only parses selected
  // by any currently-specified selection formula.
  $scope.browseCategory = function(name) {
    $scope.query = name;
    $scope.search();
  }
});

// Directive for displaying fact match counts. Used via <match-counts> tags.
app.directive('matchCounts', function() {
  return {
    scope: {
      counts: "=",
      types: "="
    },
    template: "<table class='span_fact_match'>" +
              "<thead>" +
              "<th ng-repeat='t in types'>{{t}}</th>" +
              "</thead>" +
              "<tr>" +
              "  <td ng-repeat='t in types'> " +
              "  <div class='fact_match_count'> " +
              "   {{(counts[t] != undefined) ?  counts[t].count : '-'}} " +
              "  <div class='fact_match_examples' " +
              "    ng-show='counts[t] != undefined'> " +
              "  <b>Exemplar source items</b> " +
              "   <ul> " +
              "     <li ng-repeat='example in counts[t].examples' " +
              "      class='extref'> " +
              "       <a target='_blank' " +
              "         ng-href='https://wikidata.org/wiki/{{example}}'>" +
              "          {{example}}</a> " +
              "     </li> " +
              "   </ul> " +
              "  </div> " +
              "  </div> " +
              "  </td>" +
              "</tr>" +
              "</table>"
  };
});
