# Image Generation Prompt Fine-Tuning with Llama2

## Step 1: Data Collection

To create the dataset, I utilized the GPT-4 Turbo Preview model, leveraging its capabilities through the OpenAI API. By crafting a suitable prompt, I could elicit the desired data. To safeguard sensitive API keys, I included a sample JSON file containing the results. It's noteworthy that with this approach, I could access the data directly without the need to download the model itself. For generating larger datasets, an alternative approach could involve utilizing platforms like Mostly.AI, which specializes in creating synthetic data from existing samples. This diversification in data sources is crucial for enabling productive fine-tuning across different models. Even if we don’t use Mostly.AI, creating too many data samples from the same LLM model might cause overfitting due to "clustering" samples.

## Step 2: Data Preprocessing

Upon obtaining the dataset, a simple division was performed using the Pandas library, facilitating a straightforward split. However, for more sophisticated partitioning, the `train_test_split` function from the `sklearn.model_selection` module could be employed. This flexibility allows for tailored division strategies to suit specific requirements.

## Step 3: Model Training

To prepare the model for training, a 4-bit quantized model was utilized. Subsequently, the `SFTTrainer` was employed to fine-tune the model. This process ensures that the model's parameters are adjusted to optimize performance on the specific task at hand. The difficulty here lies in determining the appropriate training strategy, as the small number of data samples may lead to overfitting after very few epochs.

## Step 4: Evaluation and Comparison

Evaluating the effectiveness of the fine-tuning process involves comparing the outputs of the original and fine-tuned models. To accomplish this, cosine similarity was calculated on the embedded representations of the model outputs. This approach enables the effective measurement of the distance between the embedded vectors, providing insight into the extent of the model's transformation. I wasn't able to use the embedding function that is being used in the real Llama 2 model due to lack of available RAM, so I used the model itself. While this might affect the real embedding similarity, I believe it still demonstrates my way of comparing text results in order to value approximation.

Furthermore, to assess the actual improvement in model performance, comparison on both the training and test sets is essential. While comparing models on the training set can indicate whether any changes have been made, evaluating them on the test set allows for a more accurate assessment of performance. By leveraging cosine similarity, the model's proximity to the true labels can be quantified, providing valuable insights into its efficacy and generalization capabilities. Additionally, by monitoring the loss and accuracy of our model during the training process, we can visualize its progress and identify potential areas for improvement, and to prevent overfitting and underfitting.

## Step 5: Bonus Analysis

The analysis of consistent behavior across different prompts' first and last layers provided valuable insights into the model's processing mechanisms. Despite the disparate nature of the input prompts, the model exhibited consistent behavior, indicative of its ability to process inputs uniformly. This consistency could be attributed to various factors, including the model's architecture, training data, and optimization process. Notably, the explanatory nature of both prompts, framed as questions with instructional guidance, significantly influenced the model's response structure.

However, the two responses I received were not satisfactory for my inquiries. This might be due to the fact that the structure of the prompts may not align well with the model's training data. If the prompts differ significantly from the types of inputs the model was trained on, it may struggle to provide relevant or coherent responses. Another potential cause could be that the model is not aligned with the specific task we are aiming to perform. In this case, if we seek informative or accurate responses to these prompts, the model's training data or fine-tuning process may not have emphasized generating such content. Additionally, the second response might be affected by the fact that the model didn’t allow me to ask harmful questions, so the ethical alignment prevented it from providing a decent response.
