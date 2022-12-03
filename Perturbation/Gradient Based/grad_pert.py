TOP_K=0.5

device = 'cuda' if torch.cuda.is_available else 'cpu'
def get_gradients(model, seq, attn_masks, token_type_ids, labels):
    """
    Get gradients by applying backward hook
    """
    extracted_grads = {}
    model.eval()
    
    def extract_grad_hook(module, grad_in, grad_out):
        
        extracted_grads['embed'] = grad_out[0]

    def add_hooks(bert_model):
        module = bert_model.embeddings.word_embeddings
        module.requires_grad_()
        module.register_backward_hook(extract_grad_hook)
        
    try:
      add_hooks(model.bert_layer)
      
    except NotImplemented:
      raise NotImplementedError
      
    prob = model(seq, attn_masks, token_type_ids)
    loss = criterion(prob, labels)
    
    loss.backward()
    return extracted_grads['embed']

def get_importance_order(model, grads, seq, attn_masks, token_type_ids, labels, is_pair=True):

    # Returns indices sorted by their importance (most - > least)
    embeds = model.bert_layer.embeddings.word_embeddings(seq)
    one_hot_grad = -torch.mul(embeds, grads)
             


    importance_order_bert1 = []
    importance_order_bert2 = []
    bert1_tok = []
    bert2_tok = []
    bert1_grads = []
    bert2_grads = []

    
    for i in range(seq.shape[0]):
        length2 = (token_type_ids[i]).sum() - 1           # length of second sentence
        length1 = (attn_masks[i]).sum() - 3 - length2     # length of first sentence
        
        bert1 = seq[i][1:1 + length1]
        bert2 = seq[i][length1 + 2 : length1 + 2 + length2]
        grads1 = one_hot_grad[i][1:1 + length1]
        grads2 = one_hot_grad[i][length1 + 2 : length1 + 2 + length2]
        bert1_tok.append(bert1)
        bert2_tok.append(bert2)
        bert1_grads.append(grads1)
        bert2_grads.append(grads2)
        order = np.argsort(grads1.sum(-1).data.cpu().numpy())
        importance_order_bert1.append(list(order))
        order = np.argsort(grads2.sum(-1).data.cpu().numpy())
        importance_order_bert2.append(list(order))

    return importance_order_bert1, importance_order_bert2, bert1_tok, bert2_tok, bert1_grads, bert2_grads


def keep_most_important_words(importance_order, bert_ids, special_ids, **kwargs):

    if 'topk' not in kwargs:
        kwargs['topk'] = 0.5

    # importance_order gives index, not the actual token_ids

    new_bert = []
    importance_order_ids = []

    for idx in importance_order:
        if bert_ids.tolist()[idx] not in special_ids:
            importance_order_ids.append(bert_ids.tolist()[idx])

    to_keep = math.ceil(len(importance_order_ids) * kwargs['topk'])
    to_keep_ids = importance_order_ids[:to_keep]


    toks = bert_ids.tolist()
    new_toks = []

    keep_ids = []
    keep_ids.extend(special_ids)
    keep_ids.extend(to_keep_ids)
    for t in toks:
        if t not in keep_ids:
            continue
        else:
            new_toks.append(t)
    new_bert.append(new_toks)

    # Wont return tensor, because we can have different length sequences
    return new_bert

def keep_one_word(importance_order, bert_ids, special_ids, **kwargs):

    # importance_order gives index, not the actual token_ids

    if 'num_words' not in kwargs:
        kwargs['num_words'] = 1

    new_bert = []
    importance_order_ids = []

    for idx in importance_order:
        if bert_ids.tolist()[idx] not in special_ids:
            importance_order_ids.append(bert_ids.tolist()[idx])

    to_keep_ids = importance_order_ids[:kwargs['num_words']]


    toks = bert_ids.tolist()
    new_toks = []

    keep_ids = []
    keep_ids.extend(special_ids)
    keep_ids.extend(to_keep_ids)
    for t in toks:
        if t not in keep_ids:
            continue
        else:
            new_toks.append(t)
    new_bert.append(new_toks)

    # Wont return tensor, because we can have different length sequences
    return new_bert


def repeat_most_important_words(importance_order, bert_ids, special_ids, **kwargs):
    # importance_order gives index, not the actual token_ids
    if 'topk' not in kwargs:
        kwargs['topk'] = 0.5

    new_bert = []

    importance_order_ids = []

    for idx in importance_order:
        if bert_ids.tolist()[idx] not in special_ids:
            importance_order_ids.append(bert_ids.tolist()[idx])

    to_keep = math.ceil(len(importance_order_ids) * kwargs['topk'])
    to_keep_ids = importance_order_ids[:to_keep]

    toks = bert_ids.tolist()
    new_toks = []

    keep_ids = []
    keep_ids.extend(special_ids)
    keep_ids.extend(to_keep_ids)
    for t in toks:
        if t not in keep_ids:
            new_toks.append(random.choice(to_keep_ids))
        else:
            new_toks.append(t)
    new_bert.append(new_toks)

    return new_bert

def replace_least_important_words(importance_order, bert_ids, special_ids, **kwargs):
    # importance_order gives index, not the actual token_ids
    if 'topk' not in kwargs:
        kwargs['topk'] = 0.5

    if 'replace_token_ids' not in kwargs:
        raise Exception

    replace_token_ids = kwargs['replace_token_ids']

    new_bert = []
    importance_order_ids = []

    for idx in importance_order:
        if bert_ids.tolist()[idx] not in special_ids:
            importance_order_ids.append(bert_ids.tolist()[idx])

    to_keep = math.ceil(len(importance_order_ids) * kwargs['topk'])
    to_keep_ids = importance_order_ids[:to_keep]

    toks = bert_ids.tolist()
    new_toks = []

    keep_ids = []
    keep_ids.extend(special_ids)
    keep_ids.extend(to_keep_ids)
    for t in toks:
        if t not in keep_ids:
            new_toks.append(random.choice(replace_token_ids))
        else:
            new_toks.append(t)
    new_bert.append(new_toks)

    return new_bert

def most_important_word_in_mention(importance_order, bert_ids1, bert_ids2, special_ids, **kwargs):
    # importance_order of premise (sent 1)
    new_bert = []
    importance_order_ids = []

    for idx in importance_order:
        if bert_ids1.tolist()[idx] not in special_ids:
            importance_order_ids.append(bert_ids1.tolist()[idx])

    most_important_tok = importance_order_ids[0]

    toks = bert_ids2.tolist()
    new_toks = []

    keep_ids = []
    keep_ids.extend(special_ids)

    for t in toks:
        if t not in keep_ids:
            new_toks.append(most_important_tok)
        else:
            new_toks.append(t)
    new_bert.append(new_toks)

    return new_bert

def load_examples(filename, input_columns, is_pair=True):
    examples = []
    header = ''
    skipped = 0
    with open(filename, 'r') as fp:
        for i, line in enumerate(fp):
            if i ==0:
                header = line.strip()
                # Header line
                continue
            inp = []
            line_split = line.strip().split(',')
            if max(input_columns) >= len(line_split):
                skipped +=1
                continue
            for col in list(input_columns):
                inp.append(line_split[col])
            examples.append((line_split, tuple(inp)))

    print('Number of examples loaded from file {} = {}'.format(filename, len(examples)))
    print('Number of examples skipped from file {} = {}'.format(filename, skipped))

    print(header)

    return examples, header

def get_function_name_dict():
    # This is a better way than using eval() or globals()
    perturbation_name_func = {
        'most_important_word_in_mention' : most_important_word_in_mention, # CopyOne
        'keep_most_important_words' : keep_most_important_words, # drop done
        'repeat_most_important_words': repeat_most_important_words, # repeat done
        'replace_least_important_words' : replace_least_important_words, # replace done
        'keep_one_word': keep_one_word,
    }

    return perturbation_name_func


def write_new_examples_to_file(filename, examples, header=None):

    fp = open(filename, 'w')
    if header is not None:
        fp.write(header + '\n')

    for ex in examples:
        output_str = ';'.join(ex)
        fp.write(output_str + '\n')

    fp.close()

from tqdm import tqdm
def generate_perturbation_gradient(dataloader, tokenizer, model, filename, input_dir, output_dir, input_columns, perturbation_func, output_filename=None, is_pair=True, on_sent2=True, **kwargs):

    examples, header = load_examples(filename, input_columns, is_pair=is_pair)

    # See if output_dir exists, otherwise create it
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if 'replace_token_ids' not in kwargs:
        kwargs['replace_token_ids'] = list(range(tokenizer.vocab_size))[100:20000]

    if output_filename is None:
        # assume file is tsv
        if 'topk' not in kwargs:
            kwargs['topk'] = 0.5
        output_filename = filename[:-4] + '_' + perturbation_func +  '_' + str(kwargs['topk']) + '.csv'

        # If is_pair task, then change filename to include arg on_sent2
        if is_pair:
            if on_sent2:
                if 'topk' not in kwargs:
                    kwargs['topk'] = TOP_K
                output_filename = filename[:-4] + '_' + perturbation_func + '_sent2' +  '_' + str(kwargs['topk']) + '.csv'
            else:
                if 'topk' not in kwargs:
                    kwargs['topk'] = TOP_K
                output_filename = filename[:-4] + '_' + perturbation_func + '_sent1' +  '_' + str(kwargs['topk']) + '.csv'

    if os.path.isfile(output_dir + '/' + output_filename):
        print('Perturbation File {} already exists !! To override, please delete that file.'.format(output_dir + '/' + output_filename))
        return output_filename

    func = get_function_name_dict()[perturbation_func]

    new_examples = []
    overall_count = 0

    if not is_pair and perturbation_func == 'most_important_word_in_mention':
        print('Perturbation : {} Invalid for single sentence. Return None '.format(perturbation_func))
        return None

    for i, (seq, attn_masks, token_type_ids, labels) in enumerate(tqdm(dataloader)):
        if i %100 == 0:
            print('Done with {0}/{1} batches'.format(i, len(dataloader)))
       

        seq, attn_masks, token_type_ids, labels = \
                seq.to(device), attn_masks.to(device), token_type_ids.to(device), labels.to(device)

        grads = get_gradients(model, seq, attn_masks, token_type_ids, labels)

        if not is_pair:
            # If a single sentence task like SST-2
            importance_order_bert, bert_tok, bert_grads = get_importance_order(model, grads, seq, attn_masks, token_type_ids, labels, is_pair=is_pair)
            for j in range(len(importance_order_bert)):
                ex = examples[overall_count]
                new_bert_ids = func(importance_order_bert[j], bert_tok[j],
                                                        tokenizer.all_special_ids, **kwargs)[0]
                new_sent = tokenizer.decode(new_bert_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                new_cols = ex[0]
                new_cols[input_columns[0]] = new_sent
                new_examples.append(new_cols)
                overall_count +=1

        else:
            #print(len(examples))
            importance_order_bert1, importance_order_bert2, bert1_tok, bert2_tok, _, _ = get_importance_order(model, grads, seq, attn_masks, token_type_ids, labels, is_pair=is_pair)
            for j in range(len(importance_order_bert2)):
                #print(overall_count)
                ex = examples[overall_count]
                new_cols = ex[0]
                if on_sent2:
                    # For on_sent2=True, copy sent1 in sent2 and then apply another perturbation (like sort)
                    if perturbation_func == 'most_important_word_in_mention':
                        new_bert_ids = func(importance_order_bert1[j], bert1_tok[j], bert2_tok[j],
                                            tokenizer.all_special_ids, **kwargs)[0]
                    else:

                        new_bert_ids = func(importance_order_bert2[j], bert2_tok[j],
                                                        tokenizer.all_special_ids, **kwargs)[0]
                    new_sent = tokenizer.decode(new_bert_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    new_cols[input_columns[1]] = new_sent
                else:
                    # For on_sent2=False, copy sent2 in sent1 and then apply another perturbation (like sort)
                    if perturbation_func == 'most_important_word_in_mention':
                        new_bert_ids = func(importance_order_bert2[j], bert2_tok[j], bert1_tok[j],
                                            tokenizer.all_special_ids, **kwargs)[0]
                    else:
                        new_bert_ids = func(importance_order_bert1[j], bert1_tok[j],
                                                        tokenizer.all_special_ids, **kwargs)[0]
                    new_sent = tokenizer.decode(new_bert_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    new_cols[input_columns[0]] = new_sent
                new_examples.append(new_cols)
                overall_count +=1


    write_new_examples_to_file(output_dir + '/' + output_filename, new_examples, header=None)
    print(len(new_examples))
    print('Successfully generated perturbations for funcion : {} with params (is_pair={}, on_sent2={}) saved to file {}'.format(
                                                                        perturbation_func, str(is_pair), str(on_sent2), output_filename))

    return output_filename
