# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Top-level PyTorch trainer.

import os
import sling
import sling.flags as flags
import sys
import time
import torch

from functools import partial

sys.path.insert(0, "sling/nlp/parser/trainer")

import train_util as utils

from train_util import mem
from train_util import now
from train_util import Resources
from pytorch_modules import Losses
from pytorch_modules import Caspar
from pytorch_modules import fstr

from corpora import Corpora

Var = torch.autograd.Variable

# Setting an explicit seed for the sake of determinism.
torch.manual_seed(1)


# Computes accuracy on the given dev set, using the given PyTorch Caspar module.
def dev_accuracy(commons_path, commons, dev_path, schema, tmp_folder, caspar):
  dev = Corpora(dev_path, commons, schema, gold=False, loop=False)
  print "Annotating dev documents", now(), mem()
  test_path = os.path.join(tmp_folder, "dev.annotated.rec")
  writer = sling.RecordWriter(test_path)
  count = 0
  start_time = time.time()

  cascade = caspar.spec.cascade
  dev_total = [0] * cascade.size()
  dev_disallowed = [0] * cascade.size()
  for document in dev:
    state, disallowed, total = caspar.forward(document, train=False)
    state.write()
    writer.write(str(count), state.encoded())
    count += 1
    if count % 100 == 0:
      print "  Annotated", count, "documents", now(), mem()
    for i, c in enumerate(disallowed):
      dev_total[i] += total[i]
      dev_disallowed[i] += c
  writer.close()
  end_time = time.time()
  print "Annotated", count, "documents in", "%.1f" % (end_time - start_time), \
      "seconds", now(), mem()
  print "Disallowed/Total leaf actions for", cascade.__class__.__name__
  for i, c in enumerate(dev_disallowed):
    print "Delegate", i, "disallowed", c, "out of", dev_total[i]

  return utils.frame_evaluation(gold_corpus_path=dev_path, \
                                test_corpus_path=test_path, \
                                commons_path=commons_path)


# A trainer reads one example at a time, till a count of num_examples is
# reached. For each example it computes the loss.
# After every 'batch_size' examples, it computes the gradient and applies
# it, with optional gradient clipping.
class Trainer:
  # Training hyperparameters.
  class Hyperparams:
    def __init__(self, args):
      # Sets various hyperparameters from 'args'.
      self.alpha = args.learning_rate
      self.batch_size = args.batch_size
      self.num_examples = args.train_steps * self.batch_size
      self.report_every = args.report_every * self.batch_size
      self.l2_coeff = args.l2_coeff
      self.gradient_clip = args.gradient_clip_norm
      self.optimizer = args.learning_method
      self.adam_beta1 = args.adam_beta1
      self.adam_beta2 = args.adam_beta2
      self.adam_eps = args.adam_eps
      self.moving_avg = args.use_moving_average
      self.moving_avg_coeff = args.moving_average_coeff
      for k, v in self.__dict__.iteritems():
        assert v is not None, "Hyperparameter %r not set" % k


    # Returns a string representation of all hyperparameters.
    def __str__(self):
      return str(self.__dict__)


  # Instantiates the trainer with the given model, optional evaluator,
  # and hyperparameters.
  def __init__(self, caspar, hyperparams, evaluator=None, \
               output_file_prefix=None):
    self.model = caspar
    self.evaluator = evaluator
    self.hyperparams = hyperparams

    if hyperparams.optimizer == "sgd":
      self.optimizer = torch.optim.SGD(
        caspar.parameters(),
        lr=self.hyperparams.alpha,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False)
    elif hyperparams.optimizer == "adam":
      self.optimizer = torch.optim.Adam(
          caspar.parameters(), lr=hyperparams.alpha, weight_decay=0, \
              betas=(hyperparams.adam_beta1, hyperparams.adam_beta2), \
              eps=hyperparams.adam_eps)
    else:
      raise ValueError('Unknown learning method: %r' % hyperparams.optimizer)

    num_params = 0
    for name, p in caspar.named_parameters():
      if p.requires_grad:
        print name, ":", p.size()
        num_params += torch.numel(p)
    print "Number of parameters:", num_params

    self.count = 0
    self.last_eval_count = 0
    self.last_update_count = 0
    self.batch_losses = None
    self._reset()

    self.checkpoint_metrics = []
    self.best_metric = None
    self.output_file_prefix = output_file_prefix

    # Exponential moving average clones.
    self.averages = {}
    if hyperparams.moving_avg:
      for name, p in caspar.named_parameters():
        if p.requires_grad:
          self.averages[name] = p.data.clone()


  # Resets the state for a new batch.
  def _reset(self):
    self.optimizer.zero_grad()
    del self.batch_losses
    self.batch_losses = Losses()


  # Processes a single given example.
  def process(self, example):
    example_losses = self.model.forward(example, train=True)
    self.batch_losses.aggregate(example_losses)
    self.count += 1
    if self.count % self.hyperparams.batch_size == 0:
      self.update()
    if self.count % self.hyperparams.report_every == 0:
      self.evaluate()


  # Clips the gradients separately for each PyTorch Parameter.
  def clip_gradients(self):
    if self.hyperparams.gradient_clip > 0.0:
      for p in self.model.parameters():
        torch.nn.utils.clip_grad_norm([p], self.hyperparams.gradient_clip)


  # Performs a gradient update.
  def update(self):
    if self.count > self.last_update_count:
      self.last_update_count = self.count
      print self.batch_losses.tostring(self.count)
      start = time.time()

      objective = self.batch_losses.average()

      # Add the regularization penalty to the objective.
      l2 = Var(torch.Tensor([0.0]))
      if self.hyperparams.l2_coeff > 0.0:
        for p in self.model.regularized_params:
          l2 += 0.5 * self.hyperparams.l2_coeff * torch.sum(p * p)
        objective += l2

      objective /= 3.0  # for parity with TF
      value = objective.data[0]

      # Compute gradients.
      objective.backward()

      # Clip them.
      self.clip_gradients()

      # Apply them.
      self.optimizer.step()

      # Done for this batch, prepare for the next one.
      self._reset()
      end = time.time()
      num_batches = self.count / self.hyperparams.batch_size

      if self.hyperparams.moving_avg:
        # Update the moving averages.
        # Use a more conservative decay factor in the first few batches.
        decay = self.hyperparams.moving_avg_coeff
        decay2 = (1.0 + num_batches) / (10.0 + num_batches)
        if decay > decay2: decay = decay2

        for name, p in self.model.named_parameters():
          if p.requires_grad and name in self.averages:
            diff = (self.averages[name] - p.data) * (1 - decay)
            self.averages[name].sub_(diff)

      print "BatchLoss after", "(%d" % num_batches, \
          "batches =", self.count, "examples):", value, \
          " incl. L2=", fstr(l2 / 3.0), \
          "(%.1f" % (end - start), "secs)", now(), mem()


  # Swaps model parameters with their moving average counterparts.
  # This just swaps pointers to data, so is very cheap.
  def _swap_with_ema_parameters(self):
    if not self.hyperparams.moving_avg: return
    for name, p in self.model.named_parameters():
      if name in self.averages:
        tmp = self.averages[name]
        self.averages[name] = p.data
        p.data = tmp


  # Runs the current model on the evaluator.
  def evaluate(self):
    if self.evaluator is not None:
      if self.count != self.last_eval_count:
        # Use average parameters if available.
        self._swap_with_ema_parameters()

        metrics = self.evaluator(self.model)
        self.checkpoint_metrics.append((self.count, metrics))
        eval_metric = metrics["eval_metric"]
        print "Eval metric after", self.count, " examples:", eval_metric

        if self.output_file_prefix is not None:
          # Record the evaluation metric to a separate file.
          if self.last_eval_count == 0:
            f = open(self.output_file_prefix + ".evals", "w")
            f.close()

          f = open(self.output_file_prefix + ".evals", "a")
          f.write("Slot_F1 after " + str(self.count) + " examples " +
                  str(eval_metric) + "\n")
          f.close()

          if self.best_metric is None or self.best_metric < eval_metric:
            self.best_metric = eval_metric

            best_flow_file = self.output_file_prefix + ".best.flow"
            self.model.write_flow(best_flow_file)
            print "Updating best flow at", best_flow_file

        self.last_eval_count = self.count

        # Swap back.
        self._swap_with_ema_parameters()


  # Trains the model using 'corpora'.
  def train(self, corpora):
    corpora.rewind()
    corpora.set_loop(True)
    for document in corpora:
      if self.count >= self.hyperparams.num_examples:
        break
      self.process(document)

    # Process the partial batch (if any) at the end, and evaluate one last time.
    self.update()
    self.evaluate()


def check_present(args, ls):
  for x in ls:
    val = getattr(args, x, None)
    assert val is not None, "%r should be present" % x
    if type(val) is str:
      assert val != "", "%r should be set" % x


def train(args):
  check_present(
      args, ["commons", "train_corpus", "output_folder", "dev_corpus"])
  resources = utils.Resources()
  resources.load(commons_path=args.commons,
                 train_path=args.train_corpus,
                 word_embeddings_path=args.word_embeddings,
                 small_spec=args.small)

  caspar = Caspar(resources.spec)
  caspar.initialize()

  tmp_folder = os.path.join(args.output_folder, "tmp")
  if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)

  evaluator = partial(dev_accuracy,
                      resources.commons_path,
                      resources.commons,
                      args.dev_corpus,
                      resources.schema,
                      tmp_folder)

  output_file_prefix = os.path.join(args.output_folder, "caspar")
  hyperparams = Trainer.Hyperparams(args)
  print "Using hyperparameters:", hyperparams

  trainer = Trainer(caspar, hyperparams, evaluator, output_file_prefix)
  trainer.train(resources.train)


if __name__ == '__main__':
  utils.setup_training_flags(flags)
  flags.define('--small',
               help='Small dimensions (for testing)',
               default=False,
               type=bool)
  flags.parse()
  train(flags.arg)
