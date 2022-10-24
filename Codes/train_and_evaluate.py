def train_and_evaluate(train_dataset,val_dataset,model, filepath, model_name, batch_size = BATCH_SIZE, learning_rate = LEARNING_RATE, epochs = EPOCHS,accumulation_steps = ACCUMULATION_STEPS):
  
  train_losses = []
  val_losses = []
    
  save_model = model_name+ '_SEED_'+ str(SEED) +'_dense_layer' +'_epoc_'+ str(epochs)+'_lr_'+ str(learning_rate)+'_b_s_'+ str(batch_size ) +'_accumulation_steps_'+ str(accumulation_steps) +'_input_type_kp_arg_topic' 
  save_predictions_name  = model_name+ '__TRAIN_PREDS_'+ 'SEED_'+ str(SEED) + '_dense_layer' +'_epoc_'+ str(epochs)+'_lr_'+ str(learning_rate)+'_b_s_'+ str(batch_size ) +'_accumulation_steps_'+ str(accumulation_steps) +'_input_type_kp_arg_topic'

  training_dataloader = DataLoader(train_dataset, batch_size )
  total_steps = len(training_dataloader) * epochs
  no_decay = ['bias', 'LayerNorm.weight']
  
  optimizer_grouped_parameters = [
                                  {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                                  {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                                  ]

  optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps = 1e-8)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

  criterion = nn.CrossEntropyLoss()
    
  model.zero_grad()

  for epoch_i in tqdm(range(epochs)):
    y_preds = []
    y_val = []
    list_of_batch_losses = []
    epoch_iterator = tqdm(training_dataloader, desc="Iteration")
    model.train()
    
    for step, batch in enumerate(epoch_iterator):
      if model_name in model_with_no_token_types:
        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        outputs = model(b_input_ids, b_input_mask)
      else:
        b_input_ids,b_token_type, b_input_mask,b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        outputs = model(b_input_ids, b_input_mask,b_token_type)
            
#       b_labels = torch.reshape(b_labels, (b_labels.shape[0], 3))
      #print(b_labels,'/n', '*'*100, outputs)
      loss = criterion(outputs, b_labels.float())
             
      list_of_batch_losses.append(loss.detach().cpu().numpy())
      run["train/batch_loss"].log(np.mean(loss.detach().cpu().numpy()))
      
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
      ypred = outputs.detach().cpu().numpy()
      b_labels = batch[-1].cpu().detach().numpy()
      ypred = np.hstack(ypred)
      y_preds.append(ypred)
      #print(ypred)
      if (step+1) % accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        model.zero_grad()


    epoch_loss = np.mean(list_of_batch_losses)
    train_losses.append(epoch_loss)
    run["train/epoch_loss"].log(epoch_loss)
    
    all_preds, all_labels , val_losses, claims = evaluate_model(val_dataset, model,  model_name, mode = 'val')
#     y_test_classes = all_labels.argmax(1)
#     y_pred_classes = all_preds.argmax(1)
#     print(all_preds, all_labels)

    f1 = custom_f1_score(np.array(all_preds), np.array(all_labels), claims)
    run["val/f1_score"].log(f1)
    print("val/f1_score", f1)
#     args = df['arg_id']
#     kps = df['key_point_id']
#     true_labels = df['label']
#     topics = df['topic']
#     stances = df['stance']
#     all_preds = []
    
#     for i in tqdm(range(len(y_preds))):
#       for p in y_preds[i]:
#         all_preds.append(p)


#     print('Train evaluation....')
    
#     pred_file = pd.DataFrame({"arg_id" : args, "key_point_id": kps, "score": all_preds})
#     args = {}
#     kps = {}

#     for arg,kp,score in zip(pred_file['arg_id'],pred_file['key_point_id'],pred_file['score']):
#         args[arg] = {}

#     for arg,kp,score in zip(pred_file['arg_id'],pred_file['key_point_id'],pred_file['score']):
#         args[arg][kp] = score

#     with open(path_predictions_folder + save_predictions_name + '_' + 'predictions.p.', 'w') as fp:
#         fp.write(json.dumps(args))
#         fp.close()
    
#     arg_df, kp_df, labels_df = load_kpm_data(path_dataset, subset="train")
#     merged_df = get_predictions(path_predictions_folder + save_predictions_name + '_' + 'predictions.p.', labels_df, arg_df, kp_df)
    
#     evaluate_predictions(merged_df,name = 'train')
    
#     _,_, val_epoch_loss = evaluate_model(val_dataset,df_val, model,  model_name, mode = 'val')
#     val_losses.append(val_epoch_loss)
    

  torch.save(model, save_model_folder +save_model+'.pt')
  run["model"].upload(save_model_folder +save_model+'.pt')

  print("Model is saved as : ",save_model)
  print("Use this to load the model")
    
  return save_model,train_losses, val_losses