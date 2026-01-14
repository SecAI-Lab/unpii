# UnPII: Unlearning Personally Identifiable Information with Quantifiable Exposure Risk

This repository will host the official implementation and artifacts for:
**UnPII: Unlearning Personally Identifiable Information with Quantifiable Exposure Risk**  
(ICSE SEIP 2026, to appear)

## Status
🚧 Code and artifacts will be released soon.  
In the meantime, please refer to the paper for details.

## Abstract
> The ever-increasing adoption of Large Language Models in critical sectors like finance, healthcare, and government raises privacy concerns regarding the handling of sensitive Personally Identifiable Information (PII) during training.
> In response, regulations such as European Union’s General Data Protection Regulation (GDPR) mandate the deletion of PII upon requests, underscoring the need for reliable and cost-effective data removal solutions.
> Machine unlearning has emerged as a promising direction for selectively forgetting data points. However, existing unlearning techniques typically apply a uniform forgetting strategy that neither accounts for the varying privacy risks posed by different PII attributes nor reflects associated business risks.
> In this work, we propose UnPII, the first PII-centric unlearning approach that prioritizes forgetting based on the risk of individual or combined PII attributes. To this end, we introduce the PII risk index (PRI), a composite metric that incorporates multiple dimensions of risk factors: identifiability, sensitivity, usability, linkability, permanency, exposability, and compliancy.
> The PRI enables a nuanced evaluation of privacy risks associated with PII exposures and can be tailored to align with organizational privacy policies. To support realistic assessment, we systematically construct a synthetic PII dataset (e.g., 1,700 PII instances) that simulates realistic exposure scenarios.
> UnPII seamlessly integrates with established unlearning algorithms, such as Gradient Ascent, Negative Preference Optimization, and Direct Preference Optimization, without modifying their underlying principles. Our experimental results demonstrate that UnPII achieves the improvements of accuracy up to 11.8%, utility up to 6.3%, and generalizability up to 12.4%, respectively, while incurring a modest fine-tuning overhead of 27.5% on average during unlearning.

## Acknowledgments
> We thank the anonymous reviewers for their constructive feedback.
> This work was partially supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korean government (MSIT; Ministry of Science and ICT) (No. RS-2024-00337414 and No. RS-2024-00437306).
> Additional support was provided by the Basic Science Research Program through the National Research Foundation of Korea (NRF), funded by the Ministry of Education of the Government of South Korea (No. RS-2025-02293072).
> Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the sponsor.
