import json
from pathlib import Path


def MMLU_accuracy(json_file:Path,):
    with open(json_file, 'r') as file:
        data = json.load(file)

    MMLU_Subject_Dict= {'hendrycksTest-college_chemistry': {'subject': 'STEM', 'samples': 100}, 'hendrycksTest-college_physics': {'subject': 'STEM', 'samples': 102}, 'hendrycksTest-formal_logic': {'subject': 'Humanities', 'samples': 126}, 'hendrycksTest-management': {'subject': 'Others', 'samples': 103}, 'hendrycksTest-virology': {'subject': 'Others', 'samples': 166}, 'hendrycksTest-high_school_physics': {'subject': 'STEM', 'samples': 151}, 'hendrycksTest-high_school_us_history': {'subject': 'Humanities', 'samples': 204}, 'hendrycksTest-high_school_government_and_politics': {'subject': 'Social_Science', 'samples': 193}, 'hendrycksTest-high_school_computer_science': {'subject': 'STEM', 'samples': 100}, 'hendrycksTest-world_religions': {'subject': 'Humanities', 'samples': 171}, 'hendrycksTest-global_facts': {'subject': 'Others', 'samples': 100}, 'hendrycksTest-us_foreign_policy': {'subject': 'Social_Science', 'samples': 100}, 'hendrycksTest-high_school_european_history': {'subject': 'Humanities', 'samples': 165}, 'hendrycksTest-college_mathematics': {'subject': 'STEM', 'samples': 100}, 'hendrycksTest-high_school_microeconomics': {'subject': 'Social_Science', 'samples': 238}, 'hendrycksTest-moral_disputes': {'subject': 'Humanities', 'samples': 346}, 'hendrycksTest-high_school_macroeconomics': {'subject': 'Social_Science', 'samples': 390}, 'hendrycksTest-international_law': {'subject': 'Humanities', 'samples': 121}, 'hendrycksTest-moral_scenarios': {'subject': 'Humanities', 'samples': 895}, 'hendrycksTest-high_school_psychology': {'subject': 'Social_Science', 'samples': 545}, 'hendrycksTest-human_sexuality': {'subject': 'Social_Science', 'samples': 131}, 'hendrycksTest-high_school_world_history': {'subject': 'Humanities', 'samples': 237}, 'hendrycksTest-computer_security': {'subject': 'STEM', 'samples': 100}, 'hendrycksTest-jurisprudence': {'subject': 'Humanities', 'samples': 108}, 'hendrycksTest-miscellaneous': {'subject': 'Others', 'samples': 783}, 'hendrycksTest-human_aging': {'subject': 'Others', 'samples': 223}, 'hendrycksTest-anatomy': {'subject': 'STEM', 'samples': 135}, 'hendrycksTest-college_medicine': {'subject': 'Others', 'samples': 173}, 'hendrycksTest-professional_medicine': {'subject': 'Others', 'samples': 272}, 'hendrycksTest-professional_psychology': {'subject': 'Social_Science', 'samples': 612}, 'hendrycksTest-logical_fallacies': {'subject': 'Humanities', 'samples': 163}, 'hendrycksTest-abstract_algebra': {'subject': 'STEM', 'samples': 100}, 'hendrycksTest-prehistory': {'subject': 'Humanities', 'samples': 324}, 'hendrycksTest-marketing': {'subject': 'Others', 'samples': 234}, 'hendrycksTest-nutrition': {'subject': 'Others', 'samples': 306}, 'hendrycksTest-clinical_knowledge': {'subject': 'Others', 'samples': 265}, 'hendrycksTest-business_ethics': {'subject': 'Others', 'samples': 100}, 'hendrycksTest-high_school_chemistry': {'subject': 'STEM', 'samples': 203}, 'hendrycksTest-philosophy': {'subject': 'Humanities', 'samples': 311}, 'hendrycksTest-public_relations': {'subject': 'Social_Science', 'samples': 110}, 'hendrycksTest-medical_genetics': {'subject': 'Others', 'samples': 100}, 'hendrycksTest-college_computer_science': {'subject': 'STEM', 'samples': 100}, 'hendrycksTest-high_school_geography': {'subject': 'Social_Science', 'samples': 198}, 'hendrycksTest-machine_learning': {'subject': 'STEM', 'samples': 112}, 'hendrycksTest-astronomy': {'subject': 'STEM', 'samples': 152}, 'hendrycksTest-conceptual_physics': {'subject': 'STEM', 'samples': 235}, 'hendrycksTest-high_school_mathematics': {'subject': 'STEM', 'samples': 270}, 'hendrycksTest-elementary_mathematics': {'subject': 'STEM', 'samples': 378}, 'hendrycksTest-electrical_engineering': {'subject': 'STEM', 'samples': 145}, 'hendrycksTest-high_school_statistics': {'subject': 'STEM', 'samples': 216}, 'hendrycksTest-professional_accounting': {'subject': 'Others', 'samples': 282}, 'hendrycksTest-professional_law': {'subject': 'Humanities', 'samples': 1534}, 'hendrycksTest-sociology': {'subject': 'Social_Science', 'samples': 201}, 'hendrycksTest-college_biology': {'subject': 'STEM', 'samples': 144}, 'hendrycksTest-high_school_biology': {'subject': 'STEM', 'samples': 310}, 'hendrycksTest-econometrics': {'subject': 'Social_Science', 'samples': 114}, 'hendrycksTest-security_studies': {'subject': 'Social_Science', 'samples': 245}}

    def calcualte_average(data, subject=None):
        sum=0
        num=0
        if subject!=None:
            for key, val in data['results'].items():
                if MMLU_Subject_Dict[key]['subject'] == subject:
                    sum += val['acc_norm']*MMLU_Subject_Dict[key]['samples']
                    num += MMLU_Subject_Dict[key]['samples']
        else:
            for key, val in data['results'].items():
                sum += val['acc_norm']*MMLU_Subject_Dict[key]['samples']
                num += MMLU_Subject_Dict[key]['samples']
        return sum/num


    print(calcualte_average(data))
    print_output = {
        "Humanities": calcualte_average(data, 'Humanities') * 100,
        "STEM": calcualte_average(data, 'STEM') * 100,
        "Social Science": calcualte_average(data, 'Social_Science') * 100,
        "Other": calcualte_average(data, 'Others') * 100,
        "Average": calcualte_average(data) * 100
    }

    formatted_output = {category: f"{value:.1f}" for category, value in print_output.items()}
    print(formatted_output)

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(MMLU_accuracy, as_positional=False)