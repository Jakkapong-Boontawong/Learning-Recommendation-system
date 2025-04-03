from flask import Flask, render_template, request
import pandas as pd
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Sample performance data for a student
performance_data = {   
    "course_name": ["Python for Beginners", "Database Fundamentals", "Advanced Virtual Reality Development"],
    "score": [55, 85, 90]
}
performance_df = pd.DataFrame(performance_data)

catalog_df = pd.read_csv("dummy_courses_50.csv")

def recommend_improvement_single_student(performance_df, catalog_df, k_lowest, top_n, preferred_media_list):

    # Sort performance data by score (lowest first) and select the k lowest courses.
    weak_topics = performance_df.sort_values(by='score', ascending=True).head(k_lowest)
    
    if weak_topics.empty:
        return []
    
    # Initialize the semantic embedding model.
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Pre-compute embeddings for all catalog courses (combining name and description).
    catalog_texts = (catalog_df['name'] + ". " + catalog_df['description']).tolist()
    catalog_embeddings = model.encode(catalog_texts, convert_to_tensor=True)
    
    # Courses the student has already taken.
    taken_courses = set(performance_df['course_name'].tolist())
    recommended_courses = []
    
    # Process each weak topic in order.
    for _, row in weak_topics.iterrows():
        weak_topic = row['course_name']
        # Try to use the course's description from catalog data.
        query_text = weak_topic
        matching = catalog_df[catalog_df['name'] == weak_topic]
        if not matching.empty:
            query_text = weak_topic + ". " + matching.iloc[0]['description']
        
        # Encode the query.
        query_embedding = model.encode(query_text, convert_to_tensor=True)
        # Compute cosine similarity scores.
        cosine_scores = util.cos_sim(query_embedding, catalog_embeddings)[0]
        similarity_scores = cosine_scores.cpu().numpy()
        
        # Build a temporary DataFrame with scores.
        temp_df = catalog_df.copy()
        temp_df['similarity_score'] = similarity_scores
        # Exclude courses already taken.
        temp_df = temp_df[~temp_df['name'].isin(taken_courses)]
        # Filter courses with low similarity.
        temp_df = temp_df[temp_df['similarity_score'] > 0.4]
        # Mark courses that match the preferred media.
        temp_df['media_match'] = temp_df['media'].apply(lambda m: 1 if m in preferred_media_list else 0)
        # Sort by similarity and media match.
        temp_df_sorted = temp_df.sort_values(by=['similarity_score', 'media_match'], ascending=False)
        recommended_courses.extend(temp_df_sorted['name'].head(top_n).tolist())
    
    # Remove duplicates while preserving order.
    seen = set()
    final_recommendations = []
    for course in recommended_courses:
        if course not in seen:
            final_recommendations.append(course)
            seen.add(course)
    
    return final_recommendations

@app.route('/', methods=['GET'])
def home():
    #recommendation function with sample parameters:
    recommendations = recommend_improvement_single_student(
        performance_df, catalog_df, k_lowest=2, top_n=3, preferred_media_list=["video", "interactive"]
    )
    return render_template("index.html", recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
