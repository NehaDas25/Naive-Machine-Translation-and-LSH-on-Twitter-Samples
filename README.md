# Work Report

## Information

- Name: <ins> DAS,NEHA </ins>
- CIN: <ins> 401457144 </ins>
- GitHub: <ins> NehaDas25 </ins>
- Email: <ins> ndas@calstatela.edu </ins>


## Features

- Not Implemented: All the codes for testing the implemented Function were provided in the assignment.
  - PART 1: 
    - Load two dictionaries mapping the English to French words
    - A training dictionary and a testing dictionary.
  - PART 2: Translation
    - Calculate transformation matrix R
    - Test cases code for the translation(K-NN algorithm).
  - PART 3: LSH And Document Search
    - 3.2: Looking up the tweets
    - 3.3: Finding the most similar tweets with LSH. Choosing the number of Planes
    - 3.6: Creating all hash tables. create_hash_id_tables() has been implemented in the assignemt. Running the cell was to create the hashes. By doing so, we ended up having several tables which have all the vectors. Given a vector, we then identified the buckets in all the tables. And then iterated over the buckets and consider much fewer vectors.
           

<br><br>

- Implemented: In the assignment, to achieve the end goal, all parts are implemented sequentially.
  - PART 1: The word embeddings data for English and French words.
    - PART 1.1 : Generate embedding and transform matrices.
      - Excercise 01 - Translating English dictionary to French by using embeddings. 
        - Implemented a function get_matrices, that takes the loaded data and returns matrices X and Y with the inputs used en_fr, french_vecs, english_vecs.
        - X_l and Y_l are lists of the english and french word embeddings. English words(keys in the dictionary) and French words(keys in the dictionary) are stored as a set under english_set and french _set.
        - Stored the french words(values of the dictionary) that are part of the english-french dictionary under french_words.
        - Looped through all english, french word pairs in the english french dictionary(en_fr).
        - stack the vectors of X_l into a matrix X and vectors of Y_l into a matrix Y using np.vstack().
        - We will use function get_matrices() to obtain sets "X_train" and "Y_train" of English and French word embeddings into the corresponding vector space models.
        - This passed all the unit-test cases.
  - PART 2: Translation.
    - PART 2.1: Translation as linear transformation of embeddings.
      - Excercise 02: Implementing translation mechanism described in this section.
       - Implemented the compute_loss() with the provided inputs X, Y, R.
       - X: a matrix of dimension (m,n) where the columns are the English embeddings.
       - Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
       - R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
       - Computed  the approximation of Y by matrix multiplying X and R (using np.dot).
       - Computed difference XR - Y.
       - Computed the squared Frobenius norm of the difference and divided it by 𝑚 which is called loss.
       - This passed the unit-test cases as well.

      - Exercise 03: Computing the gradient of loss with respect to transform matrix R.
       - Implemented the compute_gradient() with the provided inputs X, Y, R.
       - Calculated the gradient of the loss with respect to transform matrix R with the provided formula : [X^T( XR - Y) * 2/m].
       - The gradient is a matrix that encodes how much a small change in R affect the change in the loss function.
       - This passed the unit-test cases as well.


      - Exercise 04: Finding the optimal R with gradient descent algorithm.
       - Implemented a function align_embeddings() with the provided inputs as X, Y, train_steps=100, learning_rate=0.0003, verbose=True, compute_loss, compute_gradient.
       - Calculated gradient 𝑔 of the loss with respect to the matrix 𝑅.
       - Updated 𝑅 with the formula: 𝑅(new)=𝑅(old)−𝛼𝑔.
       - From this we get to know that if we make only small changes to  𝑅, we will need many steps to reach the optimum.
       - This passed the unit-test cases as well.

    - PART 2.2 Testing the translation.
      - Excercise 05 : k-Nearest neighbors algorithm.
       - Understood the k-NN algorithm. 
       - Implemented a function nearest_neighbor() with the provided inputs v, candidates, k=1, cosine_similarity.
       - v: is the vector you are going find the nearest neighbor.
       - candidates: a set of vectors where we will find the neighbors
       - k: top k nearest neighbors to find.
       - Here, for each candidate vector, cosine similarity is calculated and appended the similarity_l to the list.
       - Iterated over rows in candidates, and saved the result of similarities between current row and vector v in a python list. 
       - Sorted the similarity list and indices of the sorted list were found out and stored in sorted_ids using np.argsort.
       - Extracted the indices of the top k closest vectors in sorted form and stored in k_idx.
       - This passed the unit-test cases as well.

      - Excercise 06 : Test your translation and compute its accuracy.
       - Implemented a function test_vocabulary() with the inputs X, Y, R, nearest_neighbor.
       - Here,the prediction(pred) is X times R(pred = np.dot(X,R)) and num_correct has been initialized to 0.
       - Iterated over transformed English word embeddings and check if the closest French word vector belongs to French word that is the actual translation.
       - Obtained an index of the closest French embedding by using nearest_neighbor (with argument k=1), and compared it to the index of the English embedding you have just transformed.
       - The num_correct has been increased to 1.
       - Accuracy hae been calculated as the number correct divided by the number of rows in 'pred'.
       - This passed the unit-test cases as well. 
  - PART 3: LSH and document search
    - PART 3.1: Getting the document embeddings.
      - Excercise 07: Document Embeddings.
        - Implemented a function get_document_embedding() with  provided inputs that are tweet, en_embeddings, process_tweet.
        - This function encodes entire document as a "document" embedding. It takes in a document and a dictionary called en_embeddings.
        - This function processes the document, and looks up the corresponding embedding of each word and then added the word embedding to the running total for the document embedding and returns "doc_embedding".
        - Tested the function with the given custom tweet = "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np" in the assignment, that returns sum of all word embeddings in the tweet as array list. 
        - This passed the unit-test cases as well.
      
      - Excerise 08: Store all document vectors into a dictionary
        - Implemented the function get_document_vecs() with the inputs as all_docs, en_embeddings, get_document_embedding.
        - Here "ind2Doc_dict" is declared as an empty dictionary that stores dictionary with indices of tweets in vecs as keys and their embeddings as the values.
        - document_vec_l as an empty list that stores the document vectors.
        - Looped through all_docs, using get_document_embedding(doc,en_embeddings) doc_embedding was found and saved the document embedding into the "ind2Doc_dict" dictionary at index i.
        - Appended the document embedding to the list of document vectors that is [document_vec_l.append(doc_embedding)] and converted the list of document vectors into a 2D array using np.vstack() and returns document_vec_matrix, ind2Doc_dict.
        - This passed the unit-test cases as well.

    - PART 3.4: Getting the hash number for a vector
      - Excercise 09: Implementing hash buckets.
        - Implemented the function hash_value_of_vector() with the inputs v, planes.
        - v: vector of tweet. It's dimension is (1, N_DIMS) and planes: matrix of dimension (N_DIMS, N_PLANES) - the set of planes that divide up the region.
        - For the set of planes, dot product between the vector and the matrix containing the planes(shape: 300,10) has been calculated.So, the result of dot product will have shape of (1,10).
        - Using np.sign(), sign of the dot product has been calculated.
        - A hash vector (h) has been created by doing the following: if the element is negative, it becomes a 0, otherwise you change it to a 1. hash_value has been initialized to 0.
        - Computed the unique number for the vector by iterating over N_PLANES. Incremented the hash_value using the formula provided [ℎ𝑎𝑠ℎ=∑𝑖=0𝑁−1(2𝑖×ℎ𝑖)] and return the integer value of hash_value.
        - This passed the unit-test cases as well.

    - PART 3.5: CREATING A HASH TABLE.
      - Excercise 10:
       - Implemented the function make_hash_table() with the inputs provided are vecs, planes, hash_value_of_vector().
       - vecs: list of vectors to be hashed.
       - planes: the matrix of planes in a single "universe", with shape (embedding dimensions, number of planes).
       - Here num_of_planes is the number of columns in the planes matrix and num_buckets is 2^(number of planes).
       - After getting the num_buckets, hash_table and id_table were created as dictionary where keys are integers(0,1,2.. num_buckets) and values are empty list.
       - Looped through each vectors in 'vecs' and calculated the hash value for the vector.
       - Stored the vector into hash_table at key h,by appending the vector v to the list at key h and vector's index into id_table at key h,by appending the vector jndex i to the list at key h.
       - Returns hash_table, id_table. 
       - This passed the unit-test cases as well.


      - Excercise 11: APPROXIMATE K-NN
       - Implemented the functionapproximate_knn() with the inputs provided doc_id, v, planes_l, hash_tables, id_tables, k=1, num_universes_to_use=25, hash_value_of_vector().
       - vecs_to_consider_l has been declared as empty list that will be checked as possible nearest neighbor
       - ids_to_consider_l has been declared as empty list that stores document IDs.
       - ids_to_consider_set has been declared as empty set, for faster checking if a document ID already exists in the set.
       - Looped through the universes of planes and got set of planes from the planes_l list for a particular universe_id, hash_value of the vector for this set of planes, hash_table for this particular universe_id, document_vectors_l for the hash table where the key is the hash_value, id_table for the particular universe_id and subset of documents to consider as nearest neighbors from the id_table dictionary and is stored in new_ids_to_consider.
       - Looped through the subset of document vectors.
       - Finally returns the nearest_neighbor_ids.
       - This passed the unit-test cases as well.

        
<br><br>

- Partly implemented:
  - utils.py which contains process_tweet(),get_dict(),cosine_similarity() has not been implemented, it was provided.
  - utils_nb.py which contains process_tweet(),get_dict(),cosine_similarity() and plot_vectors() has not been implemented, it was provided.
  - w4_unittest.py was also not implemented as part of assignment to pass all unit-tests for the graded functions().

<br><br>

- Bugs
  - No bugs

<br><br>


## Reflections

- Assignment is very good. Gives a thorough understanding of the basis of Machine Translation, K-NN algorithm, Locality Sensitive Hashing and Document Search, Hash Table.


## Output

### output:

<pre>
<br/><br/>
 Out[4] - 
  The length of the English to French training dictionary is 5000
  The length of the English to French test dictionary is 1500

 Out[8] - All tests passed

 Out[14] - Expected loss for an experiment with random matrices: 8.1866
  Expected output:
      Expected loss for an experiment with random matrices: 8.1866

 Out[15] -  All tests passed 

 Out[17] - First row of the gradient matrix: [1.3498175  1.11264981 0.69626762 0.98468499 1.33828969]

  Expected output:
      First row of the gradient matrix: [1.3498175  1.11264981 0.69626762 0.98468499 1.33828969]
 Out[18] -  All tests passed
 Out[20] -
 loss at iteration 0 is: 3.7242
 loss at iteration 25 is: 3.6283
 loss at iteration 50 is: 3.5350
 loss at iteration 75 is: 3.4442
  Expected Output:
      loss at iteration 0 is: 3.7242
      loss at iteration 25 is: 3.6283
      loss at iteration 50 is: 3.5350
      loss at iteration 75 is: 3.4442

 Out[21] - All tests passed

 Out[22] -
 
 loss at iteration 0 is: 963.0146
 loss at iteration 25 is: 97.8292
 loss at iteration 50 is: 26.8329
 loss at iteration 75 is: 9.7893
 loss at iteration 100 is: 4.3776
 loss at iteration 125 is: 2.3281
 loss at iteration 150 is: 1.4480
 loss at iteration 175 is: 1.0338
 loss at iteration 200 is: 0.8251
 loss at iteration 225 is: 0.7145
 loss at iteration 250 is: 0.6534
 loss at iteration 275 is: 0.6185
 loss at iteration 300 is: 0.5981
 loss at iteration 325 is: 0.5858
 loss at iteration 350 is: 0.5782
 loss at iteration 375 is: 0.5735
    
  Expected Output
      loss at iteration 0 is: 963.0146
      loss at iteration 25 is: 97.8292
      loss at iteration 50 is: 26.8329
      loss at iteration 75 is: 9.7893
      loss at iteration 100 is: 4.3776
      loss at iteration 125 is: 2.3281
      loss at iteration 150 is: 1.4480
      loss at iteration 175 is: 1.0338
      loss at iteration 200 is: 0.8251
      loss at iteration 225 is: 0.7145
      loss at iteration 250 is: 0.6534
      loss at iteration 275 is: 0.6185
      loss at iteration 300 is: 0.5981
      loss at iteration 325 is: 0.5858
      loss at iteration 350 is: 0.5782
      loss at iteration 375 is: 0.5735

 Out[28] -
 
 [[2 0 1]
 [1 0 5]
 [9 9 9]]
 
  Expected Output:
     [[2 0 1]
      [1 0 5]
      [9 9 9]] 

 Out[29] - All tests passed

 Out[32] - 
 
 accuracy on test set is 0.557
 
  Expected Output:
      0.557
 
 Out[33] -  All tests passed

 Out[40] -
 
 array([-0.00268555, -0.15378189, -0.55761719, -0.07216644, -0.32263184])
 
  Expected output:
      array([-0.00268555, -0.15378189, -0.55761719, -0.07216644, -0.32263184])

 Out[41] - All tests passed

 Out[44] -
 
 length of dictionary 10000
 shape of document_vecs (10000, 300)
 
  Expected Output
      length of dictionary 10000
      shape of document_vecs (10000, 300)

 Out[45] -All tests passed

 Out[47] -
 
 @hanbined sad pray for me :(((

  Expected Output
      @hanbined sad pray for me :(((

 Out[48] - Number of vectors is 10000 and each has 300 dimensions.
 Out[55] -
 
 The hash value for this vector, and the set of planes at index 0, is 768

  Expected Output

      The hash value for this vector, and the set of planes at index 0, is 768

 Out[56] - All tests passed

 Out[58] -
 
 The hash table at key 0 has 3 document vectors
 The id table at key 0 has 3 document indices
 The first 5 document indices stored at key 0 of id table are [3276, 3281, 3282]

  Expected output
      The hash table at key 0 has 3 document vectors
      The id table at key 0 has 3 document indices
      The first 5 document indices stored at key 0 of id table are [3276, 3281, 3282]

 Out[59] - All tests passed

 Out[60] -
 
 working on hash universe #: 0
 working on hash universe #: 1
 working on hash universe #: 2
 working on hash universe #: 3
 working on hash universe #: 4
 working on hash universe #: 5
 working on hash universe #: 6
 working on hash universe #: 7
 working on hash universe #: 8
 working on hash universe #: 9
 working on hash universe #: 10
 working on hash universe #: 11
 working on hash universe #: 12
 working on hash universe #: 13
 working on hash universe #: 14
 working on hash universe #: 15
 working on hash universe #: 16
 working on hash universe #: 17
 working on hash universe #: 18
 working on hash universe #: 19
 working on hash universe #: 20
 working on hash universe #: 21
 working on hash universe #: 22
 working on hash universe #: 23
 working on hash universe #: 24

 Out[66] - Fast considering 77 vecs

 Out[67] -

 Nearest neighbors for document 0
 Document contents: #FollowFriday @France_Inte @PKuchly57 @Milipol_Paris for being top engaged members in my community this week :)

 Nearest neighbor at document id 51
 document contents: #FollowFriday @France_Espana @reglisse_menthe @CCI_inter for being top engaged members in my community this week :)
 Nearest neighbor at document id 2478
 document contents: #ShareTheLove @oymgroup @musicartisthere for being top HighValue members this week :) @nataliavas http://t.co/IWSDMtcayt
 Nearest neighbor at document id 105
 document contents: #FollowFriday @straz_das @DCarsonCPA @GH813600 for being top engaged members in my community this week :)

 Out[68] -

 Fast considering 77 vecs
 Fast considering 153 vecs
 All tests passed 

<br/><br/>
</pre>
