from pathlib import Path
import shutil
import os
import logging
import sys
sys.path.append('..')

from textblob import TextBlob
from pprint import pprint
from sklearn.metrics import classification_report

from transformers import AutoModelForSequenceClassification

from finbert.finbert import *
import finbert.utils as tools
import argparse

def report(df, cols=['label','prediction','logits']):
    #print('Validation loss:{0:.2f}'.format(metrics['best_validation_loss']))
    cs = CrossEntropyLoss(weight=finbert.class_weights)
    loss = cs(torch.tensor(list(df[cols[2]])),torch.tensor(list(df[cols[0]])))
    print("Loss:{0:.2f}".format(loss))
    print("Accuracy:{0:.2f}".format((df[cols[0]] == df[cols[1]]).sum() / df.shape[0]) )
    print("\nClassification Report:")
    print(classification_report(df[cols[0]], df[cols[1]], digits=3))

def args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=str, help="FinTech inference dataset.", default='fpb')
    argv = parser.parse_args()
    return argv

argv = args()


project_dir = Path.cwd()
data_dict = {'fpb':'FPB', 'fiqa':'FiQA', 'nwgi':'NWGI', 'tfns':'TFNS'}

lm_path = project_dir/'models'/'language_model'/'finbertTRC2'
cl_path = project_dir/'models'/'classifier_model_ft'/f'{data_dict[argv.data]}'
cl_data_path = project_dir/'data_vis'/f'{data_dict[argv.data]}'

# Clean the cl_path
try:
    shutil.rmtree(cl_path) 
except:
    pass

bertmodel = AutoModelForSequenceClassification.from_pretrained(lm_path,cache_dir=None, num_labels=3)


config = Config(   data_dir=cl_data_path,
                   bert_model=bertmodel,
                   num_train_epochs=4,
                   model_dir=cl_path,
                   max_seq_length = 48,
                   train_batch_size = 32,
                   learning_rate = 2e-5,
                   output_mode='classification',
                   warm_up_proportion=0.2,
                   local_rank=-1,
                   discriminate=True,
                   gradual_unfreeze=True)

finbert = FinBert(config)
finbert.base_model = 'bert-base-uncased'
finbert.config.discriminate=True
finbert.config.gradual_unfreeze=True

finbert.prepare_model(label_list=['positive','negative','neutral'])

print('preparing training data...')
train_data = finbert.get_data_json('train')
#train_data = finbert.get_data('train')

model = finbert.create_the_model()

# This is for fine-tuning a subset of the model.

freeze = 6

for param in model.bert.embeddings.parameters():
    param.requires_grad = False
    
for i in range(freeze):
    for param in model.bert.encoder.layer[i].parameters():
        param.requires_grad = False

print('start training...')
trained_model = finbert.train(train_examples = train_data, model = model)


test_data = finbert.get_data_json('test')

results = finbert.evaluate(examples=test_data, model=trained_model)

results['prediction'] = results.predictions.apply(lambda x: np.argmax(x,axis=0))

report(results,cols=['labels','prediction','predictions'])