def evaluate_model(test_dataset, model,  model_name, mode = 'train'):
    
    save_predictions_name = model_name+ '__VAL_PREDS_'+ 'SEED_'+ str(SEED) + '_dense_layer' +'_epoc_'+ str(EPOCHS)+'_lr_'+ str(LEARNING_RATE)+'_b_s_'+ str(BATCH_SIZE) +'_accumulation_steps_'+ str(ACCUMULATION_STEPS) +'_input_type_kp_arg_topic_'

    y_preds = []
    y_labels = []
    claims = []
    val_losses = []
    criterion = nn.CrossEntropyLoss()
    list_of_batch_losses = []
    
    if mode in ['train','val']:
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    with torch.no_grad():
        acc_epoch = []
        
        epoch_iterator = tqdm(test_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.eval()
            
            if model_name in model_with_no_token_types:
                b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                ypred = model(b_input_ids, b_input_mask)
            else:
                b_input_ids,b_token_type, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                ypred = model(b_input_ids, b_input_mask,b_token_type)
            
            # Store the claim to compute custom f1 metric
            for input_id in b_input_ids:
                # Find position of the last sep token and take the token before it to uniquely identify the claim
                claims.append(input_id[(input_id == tokenizer.sep_token_id).nonzero(as_tuple=True)[0][-1] - 1].detach().cpu().item())
                
#             b_labels_copy = torch.reshape(b_labels, (b_labels.shape[0], 1))
#             b_labels_copy = b_labels.copy()
            loss_batch = criterion(ypred, b_labels.float()) #check
            list_of_batch_losses.append(loss_batch.detach().cpu().numpy())
            run["val/batch_loss"].log(np.mean(loss_batch.detach().cpu().numpy()))
            
            ypred = ypred.cpu().numpy()
            b_labels = b_labels.cpu().detach().numpy()
        
            ypred_ = np.hstack(ypred.argmax(1))
            y_preds.append(ypred_)
#             print(y_preds)
            b_labels_ = np.hstack(b_labels.argmax(1))
            y_labels.append(b_labels_)           
#             print(y_labels)
    epoch_loss = np.mean(list_of_batch_losses)
    val_losses.append(epoch_loss)
    run["val/epoch_loss"].log(epoch_loss)
    
#     args = df['arg_id']
#     kps = df['key_point_id']
#     true_labels = df['label']
#     topics = df['topic']
#     stances = df['stance']
    all_preds = []
    all_labels = []

    for i in tqdm(range(len(y_preds))):
      for p1, p2  in zip(y_preds[i], y_labels[i]):
#         print(p1)
#         print(p2)
#         print(argmax)
        all_preds.append(p1)
        all_labels.append(p2)

            
#     print('Val evaluation....')
    
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
    
#     arg_df, kp_df, labels_df = load_kpm_data(path_dataset, subset="dev")
#     merged_df = get_predictions(path_predictions_folder + save_predictions_name + '_' + 'predictions.p.', labels_df, arg_df, kp_df)
    
#     evaluate_predictions(merged_df,name = 'val')

    return all_preds,all_labels, val_losses, np.array(claims)
#     return all_preds, val_losses