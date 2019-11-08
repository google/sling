import sling
import sling.flags as flags
import sling.task.workflow as workflow

flags.parse()
workflow.startup()

wf = workflow.Workflow("parser-training")

training_corpus = wf.resource(
  "local/data/corpora/caspar/train_shuffled.rec",
  format="record/document"
)

evaluation_corpus = wf.resource(
  "local/data/corpora/caspar/dev.rec",
  format="record/document"
)

word_embeddings = wf.resource(
  "local/data/corpora/caspar/word2vec-32-embeddings.bin",
  format="embeddings"
)

trainer = wf.task("caspar-trainer")

trainer.add_params({
  "learning_rate": 1.0,
  "learning_rate_decay": 0.8,
  "clipping": 1,
  "optimizer": "sgd",
  "epochs": 50000,
  "batch_size": 32,
  "rampup": 120,
  "report_interval": 500
})

trainer.attach_input("training_corpus", training_corpus)
trainer.attach_input("evaluation_corpus", evaluation_corpus)
trainer.attach_input("word_embeddings", word_embeddings)

workflow.run(wf)

workflow.shutdown()

