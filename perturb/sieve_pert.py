# for entities containing preps this pertubation is done, and for those without preps last_to_first is done.
prepositions = ['in', 'with', 'on', 'of', 'at', 'by', 'from', 'into', 'to', 'off', 'as', 'through']

def prep_replacement(entity):
  words = entity.split()
  for word in words:
    if word in prepositions:
      tries = 5
      for try_ in range(tries):
        idx = np.random.randint(0, len(prepositions))
        if word != prepositions[idx]:
          break
      pos_of_word = words.index(word)
      words[pos_of_word] = prepositions[idx]
    new_entity = ' '.join(words)

  if new_entity == entity:
     return last_to_first(entity)
  else:
      return new_entity
   

  
  
# for entities containing preps this pertubation is done, and for those without preps first_to_last is done.
prepositions = ['in', 'with', 'on', 'of', 'at', 'by', 'from', 'into', 'to', 'off', 'as', 'through']

def drop_prepositions_swap_strings(entity):
  
  words = entity.split()
  for word in words:
    if word in prepositions:
      pos_of_word = words.index(word)
      words[pos_of_word] = ''
      if pos_of_word != len(words)-1:
        words[pos_of_word-1], words[pos_of_word+1] = words[pos_of_word+1], words[pos_of_word-1]
  try:
    while True:
      words.remove('')
  except ValueError:
    pass

  new_entity = ' '.join(words)

  if new_entity == entity:
     return first_to_last(entity)
  else:
      return new_entity
  





# for words with no preps

prepositions = ['in', 'with', 'on', 'of', 'at', 'by', 'from', 'into', 'to', 'off', 'as', 'through']

def last_to_first(entity):
  words = entity.split()
  words.insert(0, words[len(words)-1])
  prep = prepositions[np.random.randint(0, len(prepositions))]
  words.insert(1, prep)
  words.pop()
  return ' '.join(words)





def first_to_last(entity):
  words = entity.split()
  prep = prepositions[np.random.randint(0, len(prepositions))]
  words.append(prep)
  words.append(words[0])
  #print(words)
  words.pop(0)
  return ' '.join(words)




dict_1 = {'1':'one', '2':'two', '3':'three', '4':'four', '5':'five', '6':'six', '7':'seven', '8':'eight', '9':'nine'}
dict_2 = {'1':'i', '2':'ii', '3':'iii', '4':'iv', '5':'v', '6':'vi', '7':'vii', '8':'viii', '9':'ix'}
dict_3 = {'1':'single', '2':'double', '3':'triple', '4':'quadruple'}       
          
import re
def how_many_forms(entity):
  entity = entity.lower().strip()
  numbers = (re.findall(r'[0-9]', entity))
  if len(numbers)==0:
    return 0
  elif int(numbers[0]) in [0]:
    return 1
  elif int(numbers[0]) in [1,2,3,4]:
    return 3
  else:
    return 2

def one_form(entity):
  return entity.replace('0', 'zero')

def two_forms(entity):
  entity = entity.lower().strip()
  number = str(re.findall(r'[0-9]', entity)[0])
  answers = []
  answers.append(entity.replace(number, dict_1[number]))
  answers.append(entity.replace(number, dict_2[number]))
  return answers

def three_forms(entity):
  entity = entity.lower().strip()
  number = re.findall(r'[0-9]', entity)[0]
  answers = []
  answers.append(entity.replace(number, dict_1[number]))
  answers.append(entity.replace(number, dict_2[number]))
  answers.append(entity.replace(number, dict_3[number]))
  return answers

def replacement_decider(row):
  if row['no of forms'] == 0:
    return row['query']
  elif row['no of forms'] == 1:
    pert_list = one_form(row['query'])
    return pert_list
  elif row['no of forms'] == 2:
    pert_list = two_forms(row['query'])
    return pert_list
  elif row['no of forms'] == 3:
    pert_list = three_forms(row['query'])
    return pert_list
  
# test1 = test.copy()
# test1['no of forms'] = test1['query'].apply(how_many_forms)

# test1['perturbed query'] = test1.apply(replacement_decider, axis=1)
# print(test1.shape)


def hyphenate(entity):
  words = entity.split()
  new_entities = []
  for i in range(1, len(words)):
    new_entities.append(' '.join(words[:i]) + '-' + ' '.join(words[i:]))
  if len(new_entities)==0:
    return entity
  return new_entities

def dehyphenate(entity):
  phrases = entity.split('-')
  new_entities = []
  for i in range(1, len(phrases)):
    new_entities.append('-'.join(phrases[:i]) + ' ' + '-'.join(phrases[i:]))
  return new_entities

def decision(entity):

  if len(entity.split('-'))>1:
    new_entities = dehyphenate(entity)
  else:
    new_entities = hyphenate(entity)
  return new_entities


# test1 = test.copy()
# test1['perturbed query'] = test1['query'].apply(decision)


# suffixation

def get_suffix(word):
  for suffix in suffix_map.keys():
    if word.endswith(suffix):
      return suffix
  return None

def suffixation(entity):
  words = entity.split()
  suffixated_entities = []
  for word in words:
    suffix = get_suffix(word)
    replace_with = None if suffix is None else suffix_map[suffix]
    if len(suffixated_entities)==0:   # first word
       if replace_with is None:
         suffixated_entities.append(word)
       else:
         for s in replace_with:
           suffixated_entities.append(word.replace(suffix, s))
    else:
      if replace_with is None:
        for i in range(len(suffixated_entities)):
          suffixated_entities[i] = suffixated_entities[i] + ' ' + word
      else:
        temp_suffixated_entities = []
        for s in replace_with:
          for suffixated_entity in suffixated_entities:
            temp_suffixated_entities.append(suffixated_entity + ' ' + word.replace(suffix, s))
        suffixated_entities = temp_suffixated_entities

  if (len(suffixated_entities)==1) and (suffixated_entities[0]==words[0]):
    return entity
  return suffixated_entities


def append_modifier(entity):
  new_entities = []
  for modifier in modifiers:
    new_entities.append(f'{entity} {modifier}')
  return new_entities

def drop_modifier(entity, idx):
  words = entity.split()
  words.pop(idx)
  return ' '.join(words)


def disorder_synonyms_replacement(entity):
  flag = False
  idx_of_modifier = 100
  words = entity.split()
  new_entities = []
  for word in words:
    if len(new_entities)==0: # first word
      if word in singular_synonyms:
        flag = True
        idx_of_modifier = words.index(word)
        idx = singular_synonyms.index(word)
        for i in range(len(singular_synonyms)):
          if i==idx:
            continue
          new_entities.append(singular_synonyms[i])

      elif word in plural_synonyms:
        flag = True
        idx_of_modifier = words.index(word)
        idx = plural_synonyms.index(word)
        for i in range(len(plural_synonyms)):
          if i==idx:
            continue
          new_entities.append(plural_synonyms[i])

      else:
        new_entities.append(word)
    
    else:
      if word in singular_synonyms:
        flag = True
        idx_of_modifier = words.index(word)
        idx = singular_synonyms.index(word)
        temp_new_entities = []
        for k in range(len(new_entities)):
          for i in range(len(singular_synonyms)):
            if i==idx:
              continue
            temp_new_entities.append(new_entities[k] + ' ' + singular_synonyms[i])
        new_entities = temp_new_entities

      elif word in plural_synonyms:
        flag = True
        idx_of_modifier = words.index(word)
        idx = plural_synonyms.index(word)
        temp_new_entities = []
        for k in range(len(new_entities)):
          for i in range(len(plural_synonyms)):
            if i==idx:
              continue
            temp_new_entities.append(new_entities[k] + ' ' + plural_synonyms[i])
            
        new_entities = temp_new_entities

      else:
        #temp_new_entities = []
        for k in range(len(new_entities)):
          new_entities[k] = new_entities[k] + ' ' + word
        #new_entities = temp_new_entities 

  if flag:
    new_entities.append(drop_modifier(entity, idx_of_modifier))
  else:
    return append_modifier(entity)
  return new_entities



plural_synonyms = ['suppresants',
'inhibitors',
'substances',
'perlongets',
'alcohols',
'poisons',
'agents',
'vaccines',
'acids',
'antigens',
'inhibitors',
'antidepressants',
'drugs',
'extracts']

singular_synonyms = ['suppresant',
'inhibitor',
'substance',
'perlonget',
'alcohol',
'poison',
'agent',
'vaccine',
'acid',
'antigen',
'inhibitor',
'antidepressant',
'drug',
'extract']

modifiers = plural_synonyms + singular_synonyms

# test1['query'] = test1['query'].apply(disorder_synonyms_replacement)
# test1.shape

import nltk
nltk.download('stopwords')
from nltk import PorterStemmer
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))


def stemmer(entity):
  words = entity.split()
  stemmed_entity = []
  for word in words:
    if word in stopwords:
      stemmed_entity.append(word)
    else:
      stemmed_token = PorterStemmer().stem(word).strip()
      if stemmed_token == '':
        stemmed_token = word
      stemmed_entity.append(stemmed_token)
  return ' '.join(stemmed_entity)
# test1['perturbed query'] = test1['query'].apply(stemmer)
# test1.shape


def composite_mention_splits_ncbi(entity):
  if len(entity.split('and or')) > 1:
    lst = entity.split('and or')
    first = lst[0]
    second = lst[1]

    first_phrase = first + ' '.join(second.split()[1:])
    second_phrase = ' '.join(first.split()[:(len(first.split())-1)]) + second
    return [first_phrase, second_phrase]
  if len(entity.split('and')) > 1:
    lst = entity.split('and')
    first = lst[0]
    second = lst[1]

    first_phrase = first + ' '.join(second.split()[1:])
    second_phrase = ' '.join(first.split()[:(len(first.split())-1)]) + second
    return [first_phrase, second_phrase]
  if len(entity.split('or')) > 1:
    lst = entity.split('or')
    first = lst[0]
    second = lst[1]
    first_phrase = first + ' '.join(second.split()[1:])
    second_phrase = ' '.join(first.split()[:(len(first.split())-1)]) + second
    return [first_phrase, second_phrase]
  return [entity]

# test1 = test.copy()
# test1['perturbed query'] = test1['query'].apply(composite_mention_splits_ncbi)



