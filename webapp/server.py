from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__, static_url_path="", static_folder="public/")


@app.route("/be-model-agents")
def model_agents():
    return render_template("be-model-agents.html")


@app.route("/predict-agent", methods=["POST"])
def predict_agent():
    avg_sales_trans = int(request.form["avg_sales_trans"].replace(",", ""))
    list_trans_count = int(request.form["list_trans_count"])
    buyer_trans_count = int(request.form["buyer_trans_count"])
    trans_count = int(list_trans_count + buyer_trans_count)
    list_sell_ratio = round(
        list_trans_count / (list_trans_count + buyer_trans_count), 2
    )
    number_of_zips = request.form["unique_zip_codes"]
    # importing the pickle
    with open("public/dt_model_agents2.obj", "rb") as fp:
        dt_model_agents = pickle.load(fp)
    #
    agent_cols = [
        "trans_count",
        "avg_sales_trans",
        "list_sell_ratio",
        "number_of_zips",
    ]
    agent_info = [
        trans_count,
        avg_sales_trans,
        list_sell_ratio,
        number_of_zips,
    ]
    #
    agent_dict = dict(zip(agent_cols, agent_info))
    agent_info = pd.DataFrame(agent_dict, index=[1])
    #
    output = dt_model_agents.predict(agent_info)
    audio = None
    if str(output[0]) == "True":
        prediction = "Agent is on track to be a top performer and produce more than $5 million in sales for 2019."
        audio = "/tada.mp3"
    else:
        prediction = "Agent is not on track to be a top performer and likely will not produce more than $5 million in sales for 2019."
        audio = "/losing.mp3"

    return render_template(
        "predict-agent.html",
        avg_sales_trans=avg_sales_trans,
        trans_count=trans_count,
        list_trans_count=list_trans_count,
        buyer_trans_count=buyer_trans_count,
        list_sell_ratio=list_sell_ratio,
        unique_zip_codes=number_of_zips,
        prediction=prediction,
        output=output,
        audio=audio,
    )


@app.route("/data-viz")
def data_viz():
    return render_template("data-viz.html")


if __name__ == "__main__":
    import waitress

    waitress.serve(app, port=5005)
