def give_dep_features(sentence):
  sent_dep_list =[]
  doc = nlp(sentence)
  for token in doc:
    sent_dep_list.append(str(token.dep_))

  return sent_dep_list