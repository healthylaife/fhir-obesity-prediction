
# <p align="center">An Interoperable ML Pipeline for Pediatric Obesity Risk Prediction using Commonly Available EHR Data</p>
  

    

## Abstract
Reliable prediction of pediatric obesity can offer a valuable resource to the providers helping them engage in timely preventive interventions before the disease is established. Many efforts have been made to develop ML-based predictive models of obesity and some studies report high predictive performances. However, no largely used clinical decision support tool based on these ML models currently exists. This study presents a novel end-to-end pipeline specifically designed for obesity prediction, which supports the entire process of data extraction, inference, and communication via an API or a user interface. By using only routinely recorded data in electronic health records (EHRs), our pipeline uses a diverse expert-curated list of medical facts to predict the risk of developing obesity. We have used input from various stakeholders, including ML scientists, providers, health IT personnel, health administration representatives, and patients throughout our design. By using the Fast Healthcare Interoperability Resources (FHIR) standard in our design procedure, we specifically target facilitating low-effort integration of our pipeline with different EHR systems. 

        
##  Installation

```
cd ./web
docker image build -t web_image .
docker run -p 443:443 -d web_image

cd ./inference
docker image build -t engine_image .
docker run -p 4000:4000  -d engine_image


https://launch.smarthealthit.org

https://localhost/launch.html
```   
