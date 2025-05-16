from approvedimports import *

def make_xor_reliability_plot(train_x, train_y):
    """ Insert code below to  complete this cell according to the instructions in the activity descriptor.
    Finally it should return the fig and axs objects of the plots created.

    Parameters:
    -----------
    train_x: numpy.ndarray
        feature values

    train_y: numpy array
        labels

    Returns:
    --------
    fig: matplotlib.figure.Figure
        figure object
    
    ax: matplotlib.axes.Axes
        axis
    """
    
    # ====> insert your code below here

    #define the range of hidden layer widths to test
    hidden_layer_width=list(range(1,11))

    #arrays to store the times training reached 100% accuracy
    successes=np.zeros(10)
    epochs=np.zeros((10,10))

    #loop over different hidden layer width
    for h_nodes in hidden_layer_width:
        for repetition in range(10):
            
            #initialize and train the MLPClassifier
            num_hidden_nodes=h_nodes
            xorMLP=MLPClassifier(
                hidden_layer_sizes=(num_hidden_nodes,),
                max_iter=1000,
                alpha=1e-4,
                solver='sgd',
                learning_rate_init=0.1,
                random_state=repetition
            )
            _=xorMLP.fit(train_x,train_y)

            #calculating traning accuracy
            training_accuracy=100*xorMLP.score(train_x,train_y)

            #if model achieved 100% accuracy, count it as a success and record number of epochs
            if training_accuracy==100:
                successes[h_nodes-1]+=1
                epochs[h_nodes-1][repetition]=xorMLP.n_iter_
            print(f"Training set accuracy: {training_accuracy}% after {xorMLP.n_iter_} iterations.")

    #calculate efficiency average epochs for successful training
    efficiency=np.zeros(10)

    for i in range(10):
        if successes[i]==0:
            efficiency[i]=1000 #penalize with max iterations if no success
        else:
            successful_epochs=epochs[i][epochs[i]>0]
            efficiency[i]=np.mean(successful_epochs)
    
    #create subplots for reliability and efficiency
    fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(12,5))

    #plot reliability 
    ax[0].plot(hidden_layer_width,successes/10*100,marker='o',color='blue',linestyle='-',markersize=6)
    ax[0].set_title("Reliability")
    ax[0].set_xlabel("Hidden Layer Width")
    ax[0].set_ylabel("Sucess Rate (%)")
    ax[0].set_xticks(hidden_layer_width)
    ax[0].set_ylim(0,100)
    ax[0].grid(True)

    #plot efficiency 
    ax[1].plot(hidden_layer_width,efficiency,marker='s',color='green',linestyle='-',markersize=6)
    ax[1].set_title("Effieciency")
    ax[1].set_xlabel("Hidden Layer Width")
    ax[1].set_ylabel("Mean Epochs")
    ax[1].set_xticks(hidden_layer_width)
    ax[1].set_ylim(0,max(efficiency)+50)
    ax[1].grid(True)

    # <==== insert your code above here

    return fig, ax

# make sure you have the packages needed
from approvedimports import *

#this is the class to complete where indicated
class MLComparisonWorkflow:
    """ class to implement a basic comparison of supervised learning algorithms on a dataset """ 
    
    def __init__(self, datafilename:str, labelfilename:str):
        """ Method to load the feature data and labels from files with given names,
        and store them in arrays called data_x and data_y.
        
        You may assume that the features in the input examples are all continuous variables
        and that the labels are categorical, encoded by integers.
        The two files should have the same number of rows.
        Each row corresponding to the feature values and label
        for a specific training item.
        """
        # Define the dictionaries to store the models, and the best performing model/index for each algorithm
        self.stored_models:dict = {"KNN":[], "DecisionTree":[], "MLP":[]}
        self.best_model_index:dict = {"KNN":0, "DecisionTree":0, "MLP":0}
        self.best_accuracy:dict = {"KNN":0, "DecisionTree":0, "MLP":0}

        # Load the data and labels
        # ====> insert your code below here
        self.data_x=np.genfromtxt(datafilename,delimiter=',')
        self.data_y=np.genfromtxt(labelfilename,delimiter=',')    
        # <==== insert your code above here

    def preprocess(self):
        """ Method to 
           - separate it into train and test splits (using a 70:30 division)
           - apply the preprocessing you think suitable to the data
           - create one-hot versions of the labels for the MLP if ther are more than 2 classes
 
           Remember to set random_state = 12345 if you use train_test_split()
        """
        # ====> insert your code below here
        self.train_x,self.test_x,self.train_y,self.test_y=train_test_split(
            self.data_x,
            self.data_y,
            test_size=0.3,
            stratify=self.data_y,
            random_state=12345
        )
        #normalise features using training mean and std
        mean=self.train_x.mean(axis=0)
        std=self.train_x.std(axis=0)
        self.train_x=(self.train_x-mean)/std
        self.test_x=(self.test_x-mean)/std

        #one hot ecnode labels for MLP
        if len(np.unique(self.train_y))>=3:
            num_classes=len(np.unique(self.train_y))
            self.train_y_mlp=np.array([
                [1 if label==i else 0 for i in range(num_classes)]
                for label in self.train_y
            ])
            self.test_y_mlp=np.array([
                [1 if label==i else 0 for i in range(num_classes)]
                for label in self.test_y
            ])
        else:
            #binary classification
            self.train_y_mlp=self.train_y
            self.test_y_mlp=self.test_y
        # <==== insert your code above here
    
    def run_comparison(self):
        """ Method to perform a fair comparison of three supervised machine learning algorithms.
        Should be extendable to include more algorithms later.
        
        For each of the algorithms KNearest Neighbour, DecisionTreeClassifer and MultiLayerPerceptron
        - Applies hyper-parameter tuning to find the best combination of relevant values for the algorithm
         -- creating and fitting model for each combination, 
            then storing it in the relevant list in a dictionary called self.stored_models
            which has the algorithm names as the keys and  lists of stored models as the values
         -- measuring the accuracy of each model on the test set
         -- keeping track of the best performing model for each algorithm, and its index in the relevant list so it can be retrieved.
        
        """
        # ====> insert your code below here
        #k-nearest neighbors test different values for k 
        for k in [1,3,5,7,9]:
            model=KNeighborsClassifier(n_neighbors=k)
            model.fit(self.train_x,self.train_y)
            acc=100*model.score(self.test_x,self.test_y)
            self.stored_models['KNN'].append(model)
            if acc>self.best_accuracy['KNN']:
                self.best_accuracy['KNN']=acc
                self.best_model_index["KNN"]=len(self.stored_models['KNN'])-1
        #dicision tree, test various depths, min_samples_split, and min_samples_leaf values
        for depth in [1,3,5]:
            for min_split in [2,5,10]:
                for min_leaf in [1,5,10]:
                    model= DecisionTreeClassifier(
                        max_depth=depth,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        random_state=12345
                    )
                    model.fit(self.train_x,self.train_y)
                    acc=100*model.score(self.test_x,self.test_y)
                    self.stored_models['DecisionTree'].append(model)
                    if acc>self.best_accuracy['DecisionTree']:
                        self.best_accuracy["DecisionTree"]=acc
                        self.best_model_index["DecisionTree"]=len(self.stored_models["DecisionTree"])-1
        #MLPClassifier: test different hidden layer sizes and activation functions
        for nodes1 in [2,5,10]:
            for nodes2 in [0,2,5]:
                for activation in ["logistic","relu"]:
                    hidden_layer=(nodes1,) if nodes2==0 else (nodes1,nodes2)
                    model=MLPClassifier(
                        hidden_layer_sizes=hidden_layer,
                        activation=activation,
                        max_iter=1000,
                        solver='sgd',
                        learning_rate_init=0.1,
                        alpha=1e-4,
                        random_state=12345
                    )
                    model.fit(self.train_x,self.train_y_mlp)
                    acc=100*model.score(self.test_x,self.test_y_mlp)
                    self.stored_models["MLP"].append(model)
                    if acc>self.best_accuracy["MLP"]:
                        self.best_accuracy["MLP"]=acc
                        self.best_model_index["MLP"]=len(self.stored_models["MLP"])-1
        # <==== insert your code above here
    
    def report_best(self) :
        """Method to analyse results.

        Returns
        -------
        accuracy: float
            the accuracy of the best performing model

        algorithm: str
            one of "KNN","DecisionTree" or "MLP"
        
        model: fitted model of relevant type
            the actual fitted model to be interrogated by marking code.
        """
        # ====> insert your code below here
        #identify the algo with the best accuracy
        best_acc=0
        best_alg=None
        for alg in ["KNN","DecisionTree","MLP"]:
            if self.best_accuracy[alg]>best_acc:
                best_acc=self.best_accuracy[alg]
                best_alg=alg
        #retrieve the best model for that algorithm
        best_model=self.stored_models[best_alg][self.best_model_index[best_alg]]
        return best_acc,best_alg,best_model
        # <==== insert your code above here
