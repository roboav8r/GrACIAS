#!/usr/bin/env python3

import gtsam

from situated_interaction.utils import normalize_vector

class DiscreteVariable():
    def __init__(self, name, var_type, method, thresh, stamp, labels, sym, idx, idx_offset, init_prob, obs_model_spec, upper_prob_limit, lower_prob_limit):
        # Admin
        self.name = name
        self.var_type = var_type
        self.update_method = method
        self.update_threshold = thresh
        self.last_updated = stamp
    
        # GTSAM
        self.var_labels = labels # TODO - make separate obs_labels for different detectors
        self.obs_labels = labels
        self.var_symbol = gtsam.symbol(sym, idx)
        self.obs_symbol = gtsam.symbol(sym, idx + idx_offset)
        self.probs = gtsam.DiscreteDistribution((self.var_symbol,len(self.var_labels)), init_prob)
        self.obs_model = gtsam.DiscreteConditional([self.obs_symbol,len(self.obs_labels)],[[self.var_symbol,len(self.var_labels)]],obs_model_spec)

        # Params
        self.upper_prob_limit = upper_prob_limit
        self.lower_prob_limit = lower_prob_limit
        
    def needs_update(self, stamp):
        return ((self.update_method=='confidence' and self.probs(self.probs.argmax()) < self.update_threshold) 
            or (self.update_method=='time' and (stamp - self.last_updated) > self.update_threshold))

    def update(self, obs_probs, stamp):
        obs = gtsam.DiscreteDistribution([self.obs_symbol,len(self.obs_labels)], obs_probs)
        likelihood = self.obs_model.likelihood(obs.argmax())
        self.probs = gtsam.DiscreteDistribution(likelihood*self.probs)

        normalized_pmf = normalize_vector(self.probs.pmf(), self.upper_prob_limit, self.lower_prob_limit)
        self.probs = gtsam.DiscreteDistribution((self.var_symbol,len(self.var_labels)),normalized_pmf)

        self.last_updated = stamp
