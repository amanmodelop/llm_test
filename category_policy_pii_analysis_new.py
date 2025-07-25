import re
import os
import time
import json
import logging
import pandas as pd
from pathlib import Path
import datetime
import numpy as np
import modelop_sdk.restclient.moc_client as moc_client
from modelop_tests.nlp_tests import examine_for_pii
from collections import defaultdict


LOG = logging.getLogger("modelop_test_wrappers.category_policy_pii_analysis")
SKIP_TESTS = []
COLUMNS: dict = {}
PII_THRESHOLD: float = 0.5
FILTER_PII = True
PII_ENTITIES = []

def find_tests_to_skip(job : dict):
    """
    Initializes the tests that should be skipped by looking at the job parameters for any present values

    :param job: The job containing the job parameters
    :return: None
    """
    global SKIP_TESTS
    if job.get("jobParameters", {}).get("skipPIIDetection", False):
        SKIP_TESTS.append("skipPIIDetection")
        LOG.warning("Skipping PII detection due to job parameters")


def find_replacement_columns(job : dict):
    """
    Finds if any columns used in the tests should be overridden from what the schema specifies

    :param job: The job containing the job parameters
    :return: None
    """
    global COLUMNS

    target_column = job.get("jobParameters", {}).get("piiAnalysisColumn", None)
    if target_column:
        COLUMNS["piiAnalysisColumn"] = target_column
        LOG.warning(f"Using column {target_column} for PII Analysis per job parameters")


#
# This is the model initialization function.  This function will be called once when the model is initially loaded.  At
# this time we can read information about the job that is resulting in this model being loaded, including the full
# job in the initialization parameter
#
# Note that in a monitor, the actual model on which the monitor is being run will be in the referenceModel parameter,
# and the monitor code itself will be in the model parameter.
#

# modelop.init
def init(init_param):
    global COLUMNS
    global SKIP_TESTS
    global PII_THRESHOLD
    global FILTER_PII
    global PII_ENTITIES
    global ACCEPTABLE_TERMS
    global INCLUDE_PII_INPUT

    COLUMNS = {}
    SKIP_TESTS = []
    job = json.loads(init_param["rawJson"])
    find_replacement_columns(job)

    # Extract input schema
    try:
        input_schemas = job["referenceModel"]["storedModel"]["modelMetaData"]["inputSchema"]
    except Exception:
        LOG.warning("No input schema found on a reference storedModel. Using base storedModel for input schema")
        input_schemas = job["model"]["storedModel"]["modelMetaData"]["inputSchema"]
    if len(input_schemas) > 1:
        LOG.error("Found more than one input schema in model definition, aborting execution.")
        raise ValueError(f"Expected only 1 input schema definition, but found {len(input_schemas)}")
    schema_df = pd.DataFrame(input_schemas[0]["schemaDefinition"]["fields"]).set_index("name")
    COLUMNS["answerColumn"] = schema_df.loc[schema_df['role'] == 'score'].index.values[0]
    COLUMNS["questionColumn"] = schema_df.loc[schema_df['role'] == 'predictor'].index.values[0]
    COLUMNS["verifiedAnswerColumn"] = schema_df.loc[schema_df['role'] == 'label'].index.values[0]
    prediction_cols = schema_df.loc[schema_df['role'] == 'prediction_date'].index.values
    if prediction_cols:
        COLUMNS["predictionDateColumn"] = prediction_cols[0]
    else:
        LOG.warning("No prediction_date role found in the schema.")
    if not COLUMNS["answerColumn"] or not COLUMNS["questionColumn"] or not COLUMNS["verifiedAnswerColumn"]:
        LOG.warning("One or more column types were not available.  Some calculations will not be possible")
    LOG.info(f"Using answer column {COLUMNS['answerColumn']}, question column {COLUMNS['questionColumn']}, verified answer column {COLUMNS['verifiedAnswerColumn']}")
    PII_THRESHOLD = job.get("jobParameters", {}).get("PII_THRESHOLD", 0.5)
    FILTER_PII = job.get("jobParameters", {}).get("FILTER_PII", True)
    PII_ENTITIES = job.get("jobParameters", {}).get("PII_ENTITIES", [])
    ACCEPTABLE_TERMS = job.get("jobParameters", {}).get("ACCEPTABLE_TERMS", [])
    INCLUDE_PII_INPUT = job.get("jobParameters", {}).get("INCLUDE_PII_INPUT", False)

    
    LOG.info(f"Using PII minimum threshold of {PII_THRESHOLD}")

# modelop.metrics
def metrics(questions_and_responses: pd.DataFrame):
    results = {}
    results.update(pii_analysis(questions_and_responses))
    results.update(category_analysis(questions_and_responses))
    yield results

def pii_analysis(questions_and_responses: pd.DataFrame):
    global COLUMNS
    global PII_THRESHOLD
    global SKIP_TESTS
    global FILTER_PII
    global PII_ENTITIES
    global ACCEPTABLE_TERMS
    global INCLUDE_PII_INPUT
    questions_and_responses=questions_and_responses.replace(r'\u0000','',regex=True)
    questions_and_responses=questions_and_responses.replace(ACCEPTABLE_TERMS,'',regex=True)
    questions_and_responses=questions_and_responses.fillna("")

    results = {}


    """
        target_column = COLUMNS.get("piiAnalysisColumn", COLUMNS.get("answerColumn"))
        questions_column = COLUMNS.get("questionColumn")

        if target_column:
            pii_result = examine_for_pii(questions_and_responses[target_column], minimum_threshold=PII_THRESHOLD, entities=PII_ENTITIES)
            if FILTER_PII:
                for result in pii_result['PII Findings']:
                    result.pop('content')
            for result in pii_result['PII Findings']:
                index = result.pop('index')
                if questions_column and INCLUDE_PII_INPUT:
                    questions = questions_and_responses[questions_column]
                    result["question"] = questions.at[index]
    """
    if not "skipPIIDetection" in SKIP_TESTS:
        target_column = COLUMNS.get("piiAnalysisColumn", COLUMNS.get("answerColumn"))
        date_column = COLUMNS.get("predictionDateColumn")
        questions_column = COLUMNS.get("questionColumn")

        if target_column:
            pii_result = examine_for_pii(questions_and_responses[target_column], minimum_threshold=PII_THRESHOLD, entities=PII_ENTITIES)
            if FILTER_PII:
                for result in pii_result['PII Findings']:
                    result.pop('content')
            for result in pii_result['PII Findings']:
                index = result.pop('index')
                if questions_column and INCLUDE_PII_INPUT:
                    questions = questions_and_responses[questions_column]
                    result["question"] = questions.at[index]        
            results.update(pii_result) 
            print("RESULTS",results)       
            # roll back the table with count info for each PII violation for the same entity and content        
            counts = defaultdict(int)
            pii_results_list=pii_result["PII Findings"]
            pii_result_only={"PII Findings":pii_results_list}
            unique_entries = {} 
            result_list = []
            for key,value in pii_result_only.items():
                for v in value:
                    k = (v["content"],v["entity_type"])
                    print("K comb",k)
                    counts[k]+= 1
                    if k not in unique_entries: 
                        unique_entries[k]= {"content":v["content"],"entity_type": v["entity_type"], "score": v["score"]}

            for key, count in counts.items():
                entry = unique_entries[key]
                result_list.append({"content":entry["content"],"type": entry["entity_type"], "score": entry["score"],"count": count})
                    
            results["PII Findings"]=result_list

            if date_column is not None and date_column in questions_and_responses:
                try:
                    dt_idx_questions_and_responses = questions_and_responses.copy()
                    dt_idx_questions_and_responses = dt_idx_questions_and_responses.set_index(pd.to_datetime(dt_idx_questions_and_responses[date_column]).dt.date)
                    dt_idx_questions_and_responses[date_column] = pd.to_datetime(dt_idx_questions_and_responses[date_column]).dt.date
                    dt_idx_questions_and_responses = dt_idx_questions_and_responses.sort_index()
                    unique_dates = dt_idx_questions_and_responses[date_column].unique()
                    array_pii_violations_day = []
                    for date in unique_dates:
                        print(f"Calculating pii violations for day {date}")
                        data_of_the_day = dt_idx_questions_and_responses.loc[[date]]
                        results_per_day = examine_for_pii(data_of_the_day[target_column], minimum_threshold=PII_THRESHOLD, entities=PII_ENTITIES)
                        array_pii_violations_day.append([str(date), float(results_per_day["num_PII_violations"])])
                    print(f"Number of PII violations over time: {array_pii_violations_day}")
                    if array_pii_violations_day:
                        pii_violations_by_day = {
                            "title": "PII Violations Over Time",
                            "x_axis_label": "Day",
                            "y_axis_label": "Number of violations",
                            "data": {
                                "num_PII_violations": array_pii_violations_day
                            }
                        }
                        results.update({
                                "pii_violations_over_time": pii_violations_by_day,
                                "firstPredictionDate": str(unique_dates.min()),
                                "lastPredictionDate": str(unique_dates.max())
                        })
                    else:
                        LOG.warning("Couldn't create line graph by date: Detected a date column, but no violations were found on any date.")
                except Exception as err:
                    err_message = str(err.args)
                    LOG.exception(f"Failed to convert date_column to standard pandas datetimes: {err_message}")
        else:
            LOG.warning("Skipped PII analysis as a column with role of score was not found in input schema")
    return results

def category_analysis(data: pd.DataFrame):
    data=data.replace(r'\u0000','',regex=True)
    data=data.fillna("")
    results={}
    print("Running the metrics function")
    data["date"]=pd.to_datetime(data["plc_mtr_timestamp"])
    data['week'] = data['date'].dt.isocalendar().week
    data['year']=data['date'].dt.isocalendar().year
    data_count=data.groupby(["year","week","que_cat_category"]).size().reset_index(name="count")
    data_count["week_date"]=[datetime.date.fromisocalendar(y,w,1).strftime("%Y-%m-%d") for y,w in zip(data_count["year"],data_count["week"])]
    data_count["count"].sum()
    graph_data={}
    for _,row in data_count.iterrows():
        value=row["que_cat_category"]
        week=row["week_date"]
        count=float(row["count"])
        if value not in graph_data:
            graph_data[value]=[]
        graph_data[str(value)].append([week,count])
    line_graph = {
    "Weekly Variation per Category": {
        "title": "Weekly Variation per Category",
        "x_axis_label": "Week",
        "y_axis_label": "Count",
        "data": graph_data
    }
    }
    data_sorted=data_count.sort_values(["que_cat_category","week_date"])
    data_sorted["percentage_change"]=data_sorted.groupby("que_cat_category")["count"].pct_change()*100
    data_sorted=data_sorted.fillna("N/A")
    generic_table = []
    for _, row in data_sorted.iterrows():
        generic_table.append({
            "Category": row['que_cat_category'],
            "Current Date": row['week_date'],
            "Current Week Count": int(row['count']),
            "% Change from Previous Week": row['percentage_change'],
        })
    data_sorted=data_sorted.sort_values(["count","week_date","que_cat_category"],ascending=False)
    results.update(line_graph)
    results.update({"Weekly Count Deviation from Previous Week":generic_table})

    return results

def main():
    job_json = json.loads(Path('test_data/example_job.json').read_text())

    init_param = {'rawJson': json.dumps(job_json)}
    init(init_param)

    questions_and_responses = pd.read_csv('test_data/example_responses.csv', quotechar='"', header=0)
    rails_tests = pd.read_csv('test_data/example_rails_test.csv', quotechar='"', header=0)

    print(json.dumps(next(metrics(questions_and_responses, rails_tests)), indent=2))

    job_json = json.loads(Path('test_data/example_job_w_parameters.json').read_text())

    init_param = {'rawJson': json.dumps(job_json)}
    init(init_param)
    print(json.dumps(next(metrics(questions_and_responses, rails_tests)), indent=2))


if __name__ == '__main__':
    main()
