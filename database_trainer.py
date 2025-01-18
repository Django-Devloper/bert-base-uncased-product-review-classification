from huggingface_hub import notebook_login
import pandas as pd 
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

data = pd.read_csv('Reviews.csv')
data = data.drop(data.columns[[0,1,2,3,4,5,7,8]],axis=1)
data = data.rename(columns ={'Score' :'label' , 'Text':'text'})
data['label'] = data['label']-1
ratings = {
    0: "Bad",
    1: "Poor",
    2: "Average",
    3: "Good",
    4: "Best"
}
data['label_name'] = data['label'].map(ratings)
data['word_per_text'] = data['text'].apply(lambda x: len(str(x).split()))
data['text'] = data['text'].fillna('')


train , test = train_test_split(data, test_size=0.3 , random_state=42 , stratify=data['label'])
test ,validate = train_test_split(test , test_size=.5 , random_state=42 , stratify=test['label'])


dataset = DatasetDict({
    'train': Dataset.from_pandas(train, preserve_index=False),
    'test': Dataset.from_pandas(test, preserve_index=False),
    'validation': Dataset.from_pandas(validate, preserve_index=False)
})

notebook_login()
dataset.push_to_hub(repo_id="djangodevloper/amazon-review-dataset-cleaned", private=False)
