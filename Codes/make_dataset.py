def make_dataset(tokenizer, tweets, claims, labels, max_len_input, model_with_no_token_types = model_with_no_token_types, model_name='roberta'):
    
    all_input_ids = []
    all_token_type_ids = []
    all_attention_masks = []
    all_labels = [] 
    
    for tweet, claim, label in zip(tweets, claims, labels) :
#         print(type(tweet))
        tweet = str(tweet)
        tweet = re.sub('[^a-zA-Z]', ' ', tweet)

        url = re.compile(r'https?://\S+|www\.\S+')
        tweet = url.sub(r'',tweet)
        
        html=re.compile(r'<.*?>')
        tweet = html.sub(r'',tweet)
        
        emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
        
        tweet = emoji_pattern.sub(r'',tweet)

        if model_name in model_with_no_token_types:

            encoded_input = tokenizer(tweet, claim, max_length = max_len_input, padding='max_length')
            all_input_ids.append(encoded_input['input_ids'])
            all_attention_masks.append(encoded_input['attention_mask'])
            #all_token_type_ids.append(encoded_input['token_type_ids'])
            all_labels.append(label)

        else :

            encoded_input = tokenizer(tweet, claim, max_length = max_len_input, padding='max_length')
            all_input_ids.append(encoded_input['input_ids'])
            all_attention_masks.append(encoded_input['attention_mask'])
            all_token_type_ids.append(encoded_input['token_type_ids'])
            all_labels.append(label)

    if model_name in model_with_no_token_types:
        all_input_ids = torch.tensor(all_input_ids).squeeze()
        all_attention_masks = torch.tensor(all_attention_masks).squeeze()
        all_labels = torch.tensor(all_labels)

        dataset = TensorDataset(all_input_ids, all_attention_masks, all_labels)

    else :
        all_input_ids = torch.tensor(all_input_ids).squeeze()
        all_token_type_ids = torch.tensor(all_token_type_ids).squeeze()
        all_attention_masks = torch.tensor(all_attention_masks).squeeze()
        all_labels = torch.tensor(all_labels) 

        dataset = TensorDataset(all_input_ids,all_token_type_ids, all_attention_masks, all_labels)

    return dataset