import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('corpus')

import torch
import numpy as np
import sys, os
import random, math


def reverse_sent(sent):

  max_tries = 10
  tries = 0

  while True:

    tok = nltk.word_tokenize(sent.lower())
    if tok[-1] == '.':
      last = tok[-1]
      new_tok = tok[:-1]
      new_tok = new_tok[::-1] # Reverse
      new_tok = new_tok + [last]
    else:
      new_tok = tok
      new_tok = new_tok[::-1] # Reverse
    new_sent = ' '.join(new_tok )
    if new_sent != sent:
        break
    elif tries >= max_tries:
        break
    else:
        tries +=1
        continue
  if new_sent==sent:
      return float('NaN')
  else:
      return new_sent

def shuffle_sent(sent):
  max_tries = 10
  tries = 0

  while True:
    tok = nltk.word_tokenize(sent.lower())
    if tok[-1] == '.':

      last = tok[-1]
      new_tok = tok[:-1]
      random.shuffle(new_tok)
      new_tok = new_tok + [last]
    else:
      random.shuffle(tok)
      new_tok = tok
      
    
    new_sent = ' '.join(new_tok )
    if new_sent != sent:
      break
    elif tries >= max_tries:
      break
    else:
      tries +=1
      continue
  if new_sent==sent:
      return float('NaN')
  else:
      return new_sent

def sort_sent(sent):
    max_tries = 10
    tries = 0

    while True:
        tok = nltk.word_tokenize(sent.lower())
        if tok[-1] == '.':
            last = tok[-1]
            new_tok = tok[:-1]
            new_tok.sort() # Sort step
            new_tok = new_tok + [last]
        else:
            new_tok = tok
            new_tok.sort() # Sort step
        new_sent = ' '.join(new_tok )
        if new_sent != sent:
            break
        elif tries >= max_tries:
            break
        else:
            tries +=1
            continue
    if new_sent==sent:
      return float('NaN')
    else:
      return new_sent

def copysort(row):
    tok = nltk.word_tokenize(row['query'])
    tok.sort()
    if ' '.join(tok) == row['candidate']:
      return float('NaN')
    else:
      return ' '.join(tok)
