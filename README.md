# Project_HSL

This is Chao's repository for the group project of HSL: Satisfaction Analysis.

Project HSL: Satisfaction Analysis is a group project that is designed, worked and delivered by three group members:

Anita Braida, Helene Ilvonen and Chao Wang


HSL: Satisfaction Analysis is a project that tries to discover the dominant feature, i.e., the key factor among Helsinki Regional Transport Authority (HSL) transportation services. In the project, we analyzed public data from the HSL satisfaction survey since 2011 and used the data to train multiple machine learning models. The result shows that, among all the features that are influencing the HSL service, the top five features are the crowdedness of the vehicle, the effectiveness of changing vehicles, fast and smooth traveling, information availability and punctuality. 

### Introduction

This project aims to provide data-driven recommendations for HSL to enhance their service quality and efficiency by focusing on the top features. Our findings will help HSL increase the number of boardings, which aligns with their 2022-2025 [development strategy](https://www.hsl.fi/en/hsl/news/news/2021/12/hsls-strategy-20222025-public-transport-back-to-an-upward-trend-through-data-and-business-driven-services-and-partnerships). And by increasing the use of public transportation, we hope to accelerate our process of transforming towards a carbon-neutral society. Studies have shown that an efficient and widely used public transportation network is associated with lower pollutants emission, improving urban air quality and reducing the carbon footprint (Gonzalez et al., 2021; Jimenez & Roman, 2021).

### Data & Methodology

In the project, we used data from multiple sources. The primary data source is the results of HSL customer satisfaction surveys, which are publicly available through [HSL official API](https://hsl.louhin.com/asty/help). The dataset consists of custom ratings from various aspects on a scale of 1-5 of HSL service (e.g, punctuality, cleanliness..) from 2011 to present. Additionally, we realized weather can be an important feature impacting the public transportation service rating, and the weather labels in the HSL survey table are not detailed enough (1 for "Rainy" and 2 for "Not rainy"). Therefore, we decided to integrate a historical weather data published by [Finnish Meteorological Institute](https://en.ilmatieteenlaitos.fi/download-observations), which offers more detailed weather data for our model to better estimate the relationships.

We obtained the raw data (646511 rows x 153 features by Sept. 2023) from the HSL's public transport customer satisfaction survey online database [Asty Web](https://hsl.louhin.com/asty/) which provides a public free [Data API](https://hsl.louhin.com/asty/help) to get the data (HSL, 2023). 

![fig:](https://lh7-us.googleusercontent.com/FR2fdKsatA2N-JcU0WkN5QiqH2lL6tc8wGd1zCZp800ZvHesXxljIpsjZpmF9xOiKRRT1GX7xlo_OSubP8nJs-3DOe_ndcjlvIUrlDnp2WJvd0SG59_U1rb08FG9-6D_0hyLN_KBVEuAT9iDTdaGtQ20J15kjY-O)

Both the original dataset and documentation is in Finnish with no English option. So we used Google translation to convert the page into English. In the documentation, it shows that the feature "K3B" is "The general rating for public transport in the HSL area". Therefore, we identified "K3B" as our target label of our training model and rest of the features can be our independent variables.

We processed the data primarily in Python using Pandas. In the dataset, each row is one survey result with each column being one question on the survey with its name in code. For example, "K1A1" stands for "Drivers/Staff serve customers in a friendly manner" and "K1A2" being "The driver knows how/ The train staff can give travel-related advice when asked". The values are users' scores for each question on a scale of 1-5, for 1 being the worst and 5 the best.

After we loaded the data, we conducted feature reduction. We investigated all the 153 features then decided to reduce the feature size in three stages.

In the first stage we removed features that are unrelated to transportation. For example, we removed "K3A6" which is "I would buy a ticket online if possible", "K3A8", "The timetable book is an important source of information..." and "K3A15" "I would like to buy and use a mobile ticket instead of a travel card" etc. Then we removed some features that are too specific on certain lines or certain times such as " K3A19", "I have previously received information about the ring road (Tikkurila-Lentoasema-Vantaankoski) that opened in summer 2015". Additionally, there are some demographic features that also get removed like "Gender", "Year of Birth" and "Home address". However, after discussion, the feature "T71": "Age group" is reserved. 

In the second stage we combined some one-hot features such as "T9","Profession" having nine features and "T21","Transportation Vehicle" with four features.

In the third stage, we decided to remove the features with over 50% missing values. As a result, we effectively reduced the number of features from 153 to 26.

As mentioned above, during the data processing we found that the weather data in the HSL survey is not clear enough as our hypothesis is that temperature and precipitation (both rain and snow) have a significant impact on customer satisfaction. Therefore, we supplemented the survey data with [external data](https://en.ilmatieteenlaitos.fi/download-observations) published by the Finnish Meteorological Institute (Institute, 2023).

![fig:](https://lh7-us.googleusercontent.com/mPcaMN0lZrs4_CVbGHorIsMQsavairv_cNEHkZ-TTrMXx3hLRbSOk8lmk7QbhWu_ImcpyM7YZ8HzTiCDT3rt6Hu109Dbhgp7Zf76pgNi-sDBTWXdZSKfrOPSzt0DYKEYmbfcebQI37MBAxcnH4UV5-THnDyjAQJ6)

For the weather dataset, we also conducted feature reduction and then merged with the HSL data. Following this, we filled missing values with the mode of the column (if they are ratings).

At last, we encoded all the values in the dataset as the last step of our data cleaning (with the data size being 623476 x 28).

![fig:](https://lh7-us.googleusercontent.com/xFGoAod7O6GB8n122c6vF2N16gu13Or9_2JZcA1RbLwk4nvbhxkY9ZUIKNrrOz2ULpqmb0Gay_Xqa3F5T5iVzWliDVRAwW8-wTek4yDeFubLy0V4CivUAX78cFjQmaOSADqgd-9nT7eeC7P_E7J5TT1UfQNIi7-H)

Before training our model, first we adopt train_test_split function to split our data into a training set (80%) and test set (20%) then normalized the data. 

As our target value is in category 1-5 (encoded as 0-4 in the model), we believe this is a multi-classification task. So the first model we used is multi-classification logistic regression from scikit-learn. 

After training, the model has an accuracy of 69.178% on the test set. The logistic regression model shows us the five most important features that are related to score 5. They are "Effective Transport Change", "Fast and Smooth Travel", "Information Availability", "Punctuality" and "Inspector's Politeness". 

![fig:](https://lh7-us.googleusercontent.com/QiaVRmEOkB_0PJS_zRwjpq81C0S_jD5xSf7TiXLPP3SUw4ZT7XIUkKLkEcAnn_1x5Lfprisq895MMZTtTpfyXKgssBrmSlFtHexVN3zjBNazgsGh0qqECKUeuyCSAMWzZv62Fhnh1px45wd_wW4xStZWcniwcYjl)

The model also shows the top five features related to the lowest score 1. They are: "Effective Transport Change", "Fast and Smooth Travel", "Information Availability", "Inspector's Politeness" and "Drivers' Friendliness". ![fig:](https://lh7-us.googleusercontent.com/UBOgbhwlBkoXwqaLv0B4khQ7beuSAgFsK_xuIZrDu2hIWqs7tBoksnXYBJeHm9Iy9MJAebHcTn4Qqh9Ro1cfpI6E-hzHIdtDXs68uO2uqV9Ri9UVvk2YbPQ3nms23gznwI6wIaWVObw0m7UsxamMyivPS8pog9Xg)

The second model we used is random forest multi-classification from scikit-learn. After training, the model's accuracy is 69% on the test set. It gives us the top seven most important features (as top 3-5 are weather features, we decided to show two more). 

They are "Boardings", "Bus/car numbers", "Weather", "Effective transportation change" and "Age group".![fig:](https://lh7-us.googleusercontent.com/5svwurJW3Y82vkBnj8OneKC-sqBaHHVfnxQtbFe9WwofTE1Df4WeEBYYl-D-9NY1gWGfszHu9JxW_pKtmWeqG1V0G0CjZcpZ9j95WzavGrNQBQJD56BZ0uUsKhek3XC9wv-otW0bjM-aja5lTGej1JzbWDv4Sqkk)

![img](https://lh7-us.googleusercontent.com/m8PJZnbvh4sdX41TRe77tXm-mBK_jzJm3hIZcbatoHQHNCSr8FfKM2UB3Dj9bq_yJ5NgoilETPF8f4iwEJc0bADfPEU4wtL8mKgOf4SM1QvgS-o0-I6LOjsewnEJogDngMTYUj6NVOYLSltp-UPKKF8)



### Discussion and Limitations

From both models we can find that "Effective transportation change" sits as one of the top features. This means that the ability to smoothly switch between vehicles is of primary importance for the users; 

Although the models provided informative features which align with the goal of the project, we found a few limitations in our project, which could be addressed in future analysis. First, there are many features which contain a high number (>50%) of missing values. This means that the dataset might not be as comprehensive as initially thought.

In a future project, it might be valuable to focus on the accessibility of the transportation network for different categories of users. Using data such as age, gender and profession in combination with the satisfaction scores might provide additional insight into the weaknesses and strengths of the service. 

### Conclusion

Therefore we strongly recommend HSL can focus on optimizing their transportation changing system as it will improve users' experience most effectively. Then HSL needs to maintain their basic transportation service in "Fast and Smooth Travel" and "Punctuality". They also need to push transportation information to the public in a timely and accessible fashion. HSL needs to remind their staff's attitude towards customers especially for drivers and inspectors.



#### Appendix:

Full code: 

https://colab.research.google.com/drive/1BJFdGe3eYM62DjuEF4LbadDL7SKVMvPI?usp=sharing

#### Works Cited

González, L., Perdiguero, J., & Sanz, À. (2021). Impact of public transport strikes on traffic and pollution in the city of Barcelona. Transportation Research Part D: Transport and Environment, 98. https://doi.org/10.1016/j.trd.2021.102952

Jiménez, F., & Román, A. (2016). Urban bus fleet-to-route assignment for pollutant emissions minimization. Transportation Research Part E: Logistics and Transportation Review, 85, 120-131. https://doi.org/10.1016/j.tre.2015.11.003

HSL. (2023). Data API. Retrieved from Asty Web: https://hsl.louhin.com/asty/help

Institute, F. M. (2023). Weather Observations. Retrieved from Finnish Meteorological Institute: https://en.ilmatieteenlaitos.fi/download-observations
