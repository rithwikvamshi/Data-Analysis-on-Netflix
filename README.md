# Netflix Data Insights

**UI Demo Video**: [Watch here](https://drive.google.com/file/d/1YMSNR7APX6J6lQ-_i4TXcXnIczLClgLI/view?usp=sharing)

## Project Overview
Netflix, one of the world’s largest streaming platforms, boasts a vast and diverse library. With over 8,000 titles, it’s essential to understand how different types of content resonate with viewers worldwide. My project dives into Netflix’s data to uncover key insights, including historical trends, popular genres, audience engagement patterns, and predictive analytics. This analysis aims to support strategic content decisions to enhance viewer satisfaction and platform growth.

## Objectives
1. Analyze Netflix’s content trends, identifying how genres, ratings, and viewership have evolved.
2. Use data visualization to interpret audience preferences and regional content strategies.
3. Predict future content success based on historical data.
4. Provide a user-friendly interactive dashboard for stakeholders to explore data insights.

## Data Description
This project utilized two main datasets:
- **Primary Netflix Dataset**: Contains fields such as title, cast, country, rating, and description.
- **External Ratings Dataset**: Additional information from Rotten Tomatoes and IMDb, providing quantitative ratings to enhance the analysis.

The datasets were combined to yield a richer dataset of 12 key fields, allowing more detailed analysis on viewer preferences, content ratings, and geographical trends.

## Data Preparation
To maintain data quality, preprocessing steps included:
- **Handling Missing Values**: Fields like "Director" and "Cast" with null values were labeled "Unknown" to ensure continuity.
- **Extracting Features**: Fields like duration and genre were standardized, and primary countries were identified to facilitate regional analysis.
- **Combining Datasets**: Titles were matched across datasets to integrate external ratings seamlessly, allowing a holistic view of Netflix’s content.

## Key Analyses and Visualizations
This project employed several visualizations to illustrate Netflix’s strategic trends and viewer engagement:

- **Growth Over Time**: Visualized the increase in content, with movies growing steadily and reflecting Netflix's strategy to cater to diverse tastes and demographics.
- **Genre and Ratings Analysis**: Box plots and word clouds revealed top genres by IMDb ratings and common themes in popular genres, such as thrillers and horror.
- **Regional Insights**: Pie charts and scatter plots highlighted Netflix’s geographic focus, with a significant title presence in North America, followed by Canada and the UK.
- **Audience Preferences**: Viewer engagement trends were mapped through IMDb scores, showing which genres were consistently rated higher, and helping identify successful content themes.

## Predictive Analytics
### Predicting Future Hits
To anticipate potential hits, I trained a machine learning model using IMDb scores as a proxy for success. After experimenting with different models, I found that **RandomForestRegressor** provided the best results due to its flexibility with categorical and numerical data. The model successfully predicted high-ranking genres and themes, proving valuable for strategic decisions around content development.

- **Model Performance**: Evaluated using Mean Squared Error (MSE) and R-squared values, with the Random Forest model achieving high accuracy, making it a reliable tool for predicting future hits on Netflix.

## UI and Interactive Dashboard
A key feature of this project is the **Interactive Dashboard**, created using the Dash library and Plotly. The dashboard includes:
- **Data Exploration Tools**: Users can view static visualizations on content trends, genres, and audience ratings.
- **Predictive Model Integration**: A predictive tool allows users to input new title data and get an IMDb score prediction.
- **Interactive Visuals**: Users can select specific years and genres to generate custom visual insights like word clouds and rating trends.
  
The dashboard provides a comprehensive interface for exploring trends and making data-driven decisions.

## Conclusion and Insights
This project highlights how Netflix's content library is structured to maximize viewer engagement across regions and demographics. Key insights include:
- **Content Growth**: Netflix’s catalog has steadily expanded, with a notable focus on movies to attract broader audiences.
- **Genre Popularity**: Certain genres, particularly dramas and thrillers, have high viewer ratings, guiding future content investments.
- **Regional Content Focus**: North American regions have a more extensive library, indicating strategic efforts in these high-market areas.
- **Predictive Success**: With reliable predictive models, Netflix can preemptively assess new titles' success potential, optimizing investment.

This project has applications beyond Netflix, providing a framework for other streaming platforms to analyze and predict content success based on viewer preferences and historical trends.

## References
- [Netflix Titles Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows?resource=download)
- [Rotten Tomatoes and IMDb Ratings Dataset](https://www.kaggle.com/datasets/ashishgup/netflix-rotten-tomatoes-metacritic-imdb)
