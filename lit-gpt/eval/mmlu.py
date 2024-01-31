import json
from pathlib import Path


def MMLU_accuracy(json_file:Path,):
    with open(json_file, 'r') as file:
        data = json.load(file)

    stem_subjects = ["hendrycksTest-abstract_algebra", "hendrycksTest-anatomy", "hendrycksTest-astronomy", 
                    "hendrycksTest-college_biology", "hendrycksTest-college_chemistry", "hendrycksTest-college_computer_science", 
                    "hendrycksTest-college_mathematics", "hendrycksTest-college_physics", "hendrycksTest-computer_security", 
                    "hendrycksTest-conceptual_physics", "hendrycksTest-electrical_engineering", "hendrycksTest-elementary_mathematics", 
                    "hendrycksTest-high_school_biology", "hendrycksTest-high_school_chemistry", "hendrycksTest-high_school_computer_science", 
                    "hendrycksTest-high_school_mathematics", "hendrycksTest-high_school_physics", "hendrycksTest-high_school_statistics", 
                    "hendrycksTest-machine_learning"]
    humanities_subjects = ["hendrycksTest-formal_logic", "hendrycksTest-high_school_european_history", 
                        "hendrycksTest-high_school_us_history", "hendrycksTest-high_school_world_history", 
                        "hendrycksTest-international_law", "hendrycksTest-jurisprudence", "hendrycksTest-logical_fallacies", 
                        "hendrycksTest-moral_disputes", "hendrycksTest-moral_scenarios", "hendrycksTest-philosophy", 
                        "hendrycksTest-prehistory", "hendrycksTest-professional_law", "hendrycksTest-world_religions"]
    social_science_subjects = ["hendrycksTest-econometrics", "hendrycksTest-high_school_geography", 
                            "hendrycksTest-high_school_government_and_politics", "hendrycksTest-high_school_macroeconomics", 
                            "hendrycksTest-high_school_microeconomics", "hendrycksTest-high_school_psychology", 
                            "hendrycksTest-human_sexuality", "hendrycksTest-professional_psychology", 
                            "hendrycksTest-public_relations", "hendrycksTest-security_studies", "hendrycksTest-sociology", 
                            "hendrycksTest-us_foreign_policy"]
    other_subjects = ["hendrycksTest-clinical_knowledge", "hendrycksTest-business_ethics", "hendrycksTest-college_medicine", 
                    "hendrycksTest-global_facts", "hendrycksTest-human_aging", "hendrycksTest-management", 
                    "hendrycksTest-marketing", "hendrycksTest-medical_genetics", "hendrycksTest-miscellaneous", 
                    "hendrycksTest-nutrition", "hendrycksTest-professional_accounting", "hendrycksTest-professional_medicine", 
                    "hendrycksTest-virology"]

    def calculate_average_metrics(subjects, data):
        acc_values = [data['results'][subject]['acc'] for subject in subjects if subject in data['results']]
        acc_norm_values = [data['results'][subject]['acc_norm'] for subject in subjects if subject in data['results']]
        average_acc = sum(acc_values) / len(acc_values) if acc_values else 0
        average_acc_norm = sum(acc_norm_values) / len(acc_norm_values) if acc_norm_values else 0
        return average_acc, average_acc_norm

    average_acc_stem, average_acc_norm_stem = calculate_average_metrics(stem_subjects, data)
    average_acc_humanities, average_acc_norm_humanities = calculate_average_metrics(humanities_subjects, data)
    average_acc_social_science, average_acc_norm_social_science = calculate_average_metrics(social_science_subjects, data)
    average_acc_other, average_acc_norm_other = calculate_average_metrics(other_subjects, data)

    total_subjects = stem_subjects + humanities_subjects + social_science_subjects + other_subjects
    overall_average_acc, overall_average_acc_norm = calculate_average_metrics(total_subjects, data)

    print_output = {
        "Humanities": average_acc_norm_humanities * 100,
        "STEM": average_acc_norm_stem * 100,
        "Social Science": average_acc_norm_social_science * 100,
        "Other": average_acc_norm_other * 100,
        "Average": overall_average_acc * 100
    }
    formatted_output = {category: f"{value:.1f}" for category, value in print_output.items()}
    print(formatted_output)
    
    

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(MMLU_accuracy, as_positional=False)