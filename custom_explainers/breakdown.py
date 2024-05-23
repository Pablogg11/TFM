import dalex as dx
import numpy as np
import pandas as pd

class Breakdown:
    def __init__(self, model, data):
        self.model = model
        self.ex = dx.Explainer(self.model, data[0], data[1].ravel())

    def explain(self,  test_set):
        test_set = test_set.to_numpy()
        att = []
        for x in test_set:
            bd = self.ex.predict_parts(new_observation = x, type = "break_down")
            result = bd.result.sort_values(by='variable_name')
            contribution = result.contribution.to_numpy()
            contribution = contribution[1:-1]
            att.append(contribution)

        return np.array(att)
    
class IBreakdown:
    def __init__(self, model, data):
        self.model = model
        self.ex = dx.Explainer(self.model, data[0], data[1].ravel())


    def explain(self,  test_set):
        test_set = test_set.to_numpy()
        att = []
        for x in test_set:
            bd = self.ex.predict_parts(new_observation = x, type = "break_down_interactions")
            result = bd.result[["variable_name","contribution"]].values.tolist()
            result = result[1:-1]
            aux = []
            for feat in result:
                if ':' in feat[0]:
                    value = feat[1]
                    feats = feat[0].split(':')
                    aux.append([feats[0],value])
                    aux.append([feats[1],value])
                else:
                    aux.append(feat)
            contribution = sorted(aux, key=lambda x: x[0])
            contribution = [x[1] for x in contribution]
            att.append(contribution)
            
        return np.array(att)
    
    # class IBreakdown:
    #     def __init__(self, model, data):
    #         self.model = model
    #         self.ex = dx.Explainer(self.model, data[0], data[1].ravel())

    #     def explain(self,  test_set):
    #         test_set = test_set.to_numpy()
    #         att = []
    #         for x in test_set:
    #             bd = self.ex.predict_parts(new_observation = x, type = "break_down_interactions", interaction_preference = 0)
    #             result = bd.result.sort_values(by='variable_name')
    #             contribution = result.contribution.to_numpy()
    #             contribution = contribution[1:-1]
    #             att.append(contribution)
                
    #         print(np.array(att))
    #         return np.array(att)
