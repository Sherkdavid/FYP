# FYP
Code file for final year research project investigating solar photovoltaic prediction modelling using weather feature data. Functions perform
relevant pre-processing, data interrogation, k-fold cross validation, visualization and parameter optimization of algorithm. A portfolio of algorithms are
used to make predictions in the following process;

- Day instance identified for prediction
- Instance evaluated with K Nearest Neighbour to identify a subset of data nearer to the instance
- Linear regression model trained on the subset of similar data
- Instance power output predicted using regression

This approach is informed by the assertion discovered in the research phase that similar days will prove more insightful in making
predictions where climate is determinent of the target.
