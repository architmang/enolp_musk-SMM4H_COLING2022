def give_padding(sent):

  var_len = len(sent)
  padding_len = 349-var_len
  if (padding_len>0):
    padding = [0.0]*padding_len
    sent = sent + padding
  else:
    sent = sent[:349]
  return sent