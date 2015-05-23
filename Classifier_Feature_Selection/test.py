import pickle

classif_predictions=pickle.load(open(current_dir+ '/Predictions_Classifier/Log'+ "_proba_val_N_feat="+ str(N_feat) + "_N_sentences="+ str(N_sentence)+ ".p", "rb"))

