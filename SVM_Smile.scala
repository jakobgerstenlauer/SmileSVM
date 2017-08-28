// Creating synthetic data for a classification problem
// Here I simulate a Gaussian Mixture. 
// Compare the R code in the sgmcmc vignette:
// https://cran.r-project.org/web/packages/sgmcmc/vignettes/gaussMixture.html) .

import breeze.linalg._
import breeze.numerics._
import breeze.math._
import breeze.linalg.NumericOps 
import breeze.linalg.operators 
import breeze.stats.distributions._

//Simulate a Gaussian Mixture
//The number of observations (cases/ instances):
val N = 5000
val N_train = floor(N * 0.5).toInt
//The number of features (inputs/ variables):
val d = 20

// Set locations of two modes, theta1 and theta2
val theta1 = DenseVector.rand(d)
val theta2 = DenseVector.rand(d)

//Randomly allocate observations to each class (0 or 1)
val z = DenseVector.rand(N).map(x=>2*x).map(x=>floor(x))

//empty data matrix
val X = DenseMatrix.zeros[Double](N,d)

val mvn1 = breeze.stats.distributions.MultivariateGaussian(theta1, diag(DenseVector.fill(d){1.0}))
val mvn2 = breeze.stats.distributions.MultivariateGaussian(theta2, diag(DenseVector.fill(d){1.0}))

// Simulate each observation depending on the component its been allocated
// create all inputs (predictors)
var x = DenseVector.zeros[Double](d) 
var X_2 = Array.ofDim[Double](N, d)

for(i <- 0 to (N-1)){
  if ( z(i) == 0 ) {
    x = mvn1.sample()
  }else{
    x = mvn2.sample()
  }
  //matrix assignment to column
  X (i,::) := DenseVector(x.toArray).t
  var j = 0
  for (xe <- x.toArray){
    X_2(i)(j)=xe
    j=j+1
  }
}

//Calculate the variance of the synthetic data set. The empirical variance should be very close to 1.0 because that is how the generating multivariate normal distribution has been specified.

import breeze.stats.{variance}
println("The variance of columns:")
println(variance(X(::,*)))

// Train the Support Vector Machine with a Gaussian kernel on the training data
import smile.classification.{SVM,SoftClassifier}
import smile.math.kernel.{GaussianKernel}

val svm = new SVM[Array[Double]](new GaussianKernel(1.5), 2.0, 2)
val label = z.toArray.map(x=>x.toInt)

//How does slicing of an array of arrays work?
var M = Array.ofDim[Double](2, 2)
M(0)(0)=1
M(0)(1)=2
M(1)(0)=3
M(1)(1)=4
M.slice(0,1)

val X_train = X_2.slice(0, N_train)
val label_train = label.slice(0, N_train  )
val X_test = X_2.slice(N_train, N)
val label_test = label.slice(N_train, N)

svm.learn(X_train, label_train)
svm.finish()

//Calculate the training error 
val falsePredictionsTrain = X_train
    .map(x => svm.predict(x))
    .zip(label_train)
	.map(x => if (x._1 == x._2) 0 else 1 )
	.reduce(_ + _)


//Assess the generalization error of the SVM model on the test data.
val falsePredictionsTest = X_test
    .map(x => svm.predict(x))
    .zip(label_test)
	.map(x => if (x._1 == x._2) 0 else 1 )
	.reduce(_ + _)

//Now, let's do a more systematic approach using the smile cross-validation function.
import smile.validation.{Validation, ClassificationMeasure, Accuracy}
import smile.validation.Validation.cv
import smile.classification.ClassifierTrainer
import smile.classification.SVM.Trainer

val trainer = new Trainer(new GaussianKernel(1.5), 2.0, 2)
/**
* Cross validation of a classification model.
* 
* @param <T> the data type of input objects.
* @param k k-fold cross validation.
* @param trainer a classifier trainer that is properly parameterized.
* @param x the test data set.
* @param y the test data labels.
* @param measure the performance measure of classification.
* @return the test results with the same size of order of measures
*/
//   public static <T> double cv(int k, ClassifierTrainer<T> trainer, T[] x, int[] y, ClassificationMeasure measure) {
val crossValidationMeasure = cv(10, trainer, X_2, label, new Accuracy)


// We fitted a 10-fold cross-validation model with the same SVM specification. Now, let's run a grid search to do a systematic parameter tuning of the kernel parameter sigma and the SVM parameter C.

var maxAccuracy = 0.0
var sigma = -1.0
var C = -1.0
for(i <- 1 until 10){
  for(j <- 1 until 10){
      val trainer = new Trainer(new GaussianKernel(i*0.2), j*0.3, 2)
      val crossValidationMeasure = cv(10, trainer, X_2, label, new Accuracy)
      if (crossValidationMeasure > maxAccuracy){
          maxAccuracy = crossValidationMeasure
          sigma = i*0.2
          C = j*0.3
      }
   }
}

//Calculate the estimated class probabilities of the best model.
val svm = new SVM[Array[Double]](new GaussianKernel(sigma), C, 2)
println(svm.hasPlattScaling())
var posteriori = Array.ofDim[Double](N, 2)
svm.trainPlattScaling(X_2, label)
println(svm.hasPlattScaling())
//Calculate the posterior class probabilities
for (i <- 0 until N){
  svm.predict(X_2(i),posteriori(i))
} 

//Check if there are any observations with intermediate posterior probabilities 
posteriori.map(x => if (x(0) < 0.8 && x(1) < 0.8) 1 else 0).reduce(_ + _)

