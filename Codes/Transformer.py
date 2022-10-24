# Use this Class only for Bert base and Bert large model

class Transformer(nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()
        
        #Instantiating Pre trained model object 
        self.model_layer = AutoModel.from_pretrained(model_path)
        
        #Layers
        # the first dense layer will have 768 neurons if base model is used and 
        # 1024 neurons if large model is used

        self.dense_layer_1 = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.4)
        self.dense_layer_2 = nn.Linear(256, 128)
        self.dropout_2 = nn.Dropout(0.4) 
#         self.cls_layer = nn.Linear(128, 1, bias = True)
        self.cls_layer = nn.Linear(128, 3, bias = True)
        
        self.Softmax = nn.Softmax(dim=1) #check dim

    def forward(self,input_ids, attention_masks, token_type_ids):

        pooled_output = self.model_layer(input_ids=input_ids, attention_mask=attention_masks,token_type_ids = token_type_ids).pooler_output
        
        x = self.dense_layer_1(pooled_output)
        x = self.dropout(x)
        x_1 = self.dense_layer_2(x)
        x_2 = self.dropout_2(x_1)
        
        logits = self.cls_layer(x_2)
        output = self.Softmax(logits)  # add logits with BCEloss for better numerical stability(on the downside, have to pass labels)

        return output