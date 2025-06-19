# add duplicate detection, maybe stopword removal...
from sentence_transformers import SentenceTransformer, util

# relevance
def get_feedback_relevance():

model = SentenceTransformer('all-MiniLM-L6-v2')
product_description = "SaaS tool for managing customer feedback"
comment = "I hate how this app crashes on my old Android phone."

embedding1 = model.encode(product_description, convert_to_tensor=True)
embedding2 = model.encode(comment, convert_to_tensor=True)

similarity = util.cos_sim(embedding1, embedding2)
if similarity > 0.5:  # Tune threshold
    print("Relevant")