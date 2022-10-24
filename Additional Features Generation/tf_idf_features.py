def give_tfidf(df,topics):
  big_list = []
  
  for i in topics:
    tfidf_vectorizer = TfidfVectorizer(stop_words='english',min_df = 10, max_df = 0.5, ngram_range=(1,2))
    topic_1 = tfidf_vectorizer.fit_transform(list(df[df['claim'] == i]['sentence'].values))
    topic_1_matrix = topic_1.toarray()

    topic_1_list =[]
    for j in topic_1_matrix:
      topic_1_list.append(list(j))
    
    big_list = big_list +topic_1_list
  
  return big_list