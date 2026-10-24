Daily Progress Log
02/10
Completed: 
- explored data using ChatGPT and inspecting datasets
Findings:
- can't match up occupancy and environmental data directly with same dates
- can't match up rooms between datasets either
- limited usable environmental data available
- from May-Dec 2024, only one enviro sensor has measurements from May-June
Decisions:
- go in direction of identifying operating patterns

13/10
Completed: 
- made final decision on project topic
- set up VS code and github
Findings:
- both data sets need to be used
- time frame for enviro data is 1st May - 12th June
Decisions:
- Project question: can environmental data be used to predict building occupancy?

17/10
Completed: 
- finish setting up github and project documentation
Findings:
Decisions:

18/10
Completed: 
- uploaded raw data files
- cleaned enviro data from 1st may-12th june for sensor 6012002000869
Findings:
- enviro data too sparse
Decisions:
- change project to just occupancy pattern identification

19/10
Completed:
- re cleaned occupancy data in full
- checked dates
- checked what headcount represents
- aggregated occupancy data into hourly averages
Findings:
- dates range more than maytodec2024
- headcount: number of people in space in that moment
Decisions:

22/10
Completed:
- analysed weekly patterns, results in graph form
Findings:
- predictions can be made from this data, there are clear patterns
Decisions:
- split data: 70% training, 15% validation, 15% testing

23/10
Completed:
- analysed weekly data 
- predicted occupancy and analysed results using different methods
- 2nd experiment using only spaces 1,2,4,5
- checked times of different models/number of trees for optimisation
- checked sensitivity
Findings:
- space 3 ends observation on 30th december 2024
- trends change over time - weekly predictons in 2023 won't translate the same to 2026
- limitations: space 3 not reliable after 2024, random forest performed badly in validation testing but better in testing, suggesting its accuracy varies over time
- patterns don't remain stable
- weekly pattern prediction has lowest validation MAE so is more defensible than random forest. has middle test MAE and RMSE
- random forest more sensitive to differences between time periods, weekly more stable
- smaller random forest - more efficient, more complex doesn't mean better accuracy, weekly model best
- performed better with 50% coverage than 80%
Decisions:
- exclude space 3 from predictions
- focus only on weekly patterns 
- go with weekly model - lowest validation mae and quicker time
- stick with 80% coverage and suggest 50% as improvement

24/10
Completed:
- README file completed
- file/function documentation updated 
- final checks on github done 
- test files created and checked
Findings:
- everything in order
Decisions:
- ready to write up project documentation