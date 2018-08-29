V2 : CNN+RNN(20*3072 -> 3072) xavier init. (Fail : <20%)
V3 : CNN+RNN(20*3072 -> 20*3072) xavier init.
V4 : V3 with random normal init.
V5 : V2 with dropout 0.7(keep_prob)
V6 : V3 with dropout 0.7(keep_prob)
V7 : CNN each of 20 frames and combine with weights. (partial success : 30~40%)
V8 : CNN only 10th frame (Fail : <25%)
