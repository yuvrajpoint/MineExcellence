import React, { useState } from "react";
import axios from "axios";
const PredictForm = () => {
  const [graphs, setGraphs] = useState(null);
  const [inputs, setInputs] = useState({
    'Hole dia. [mm]': 115,
    'Hole depth [m]': 10.5,
    'No. of holes': 5,
    'Avg. Burden [m]': 4.5,
    'Avg.Spacing [m]': 7.5,
    'Avg. top stemming length [m]': 3,
    'Avg. charge/hole [kg]': 58.65,
    'Total charge [kg]': 293.25,
    'Max.charge delay [kg]': 59,
    'distance': 200,
    'Pit': 1,
    'bno': 12,
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setInputs({ ...inputs, [e.target.name]: parseFloat(e.target.value) });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const res = await axios.post("http://127.0.0.1:5000/predict", inputs);
    setResult(res.data.ppv);
    setGraphs(res.data.graph);
  };

  return (
    <div className="mb-8 p-4 rounded-xl bg-gray-800 shadow-lg">
      <h2 className="text-2xl mb-4 text-orange_accent">Predict PPV</h2>
      <form onSubmit={handleSubmit} className="grid grid-cols-3 gap-4">
        {Object.keys(inputs).map((key) => (
          <div key={key}>
            <label className="block text-sm">{key}</label>
            <input
              type="number"
              name={key}
              value={inputs[key]}
              onChange={handleChange}
              className="w-full p-2 r``ounded bg-gray-700 text-white"
            />
          </div>
        ))}
      </form>
      <button
        onClick={handleSubmit}
        className="mt-4 px-4 py-2 bg-orange_accent rounded hover:bg-orange-600 transition"
      >
        Predict
      </button>

      {result !== null && (
        <div className="mt-4 text-xl">
          Predicted PPV:{" "}
          <span className="text-orange_accent">{result.toFixed(2)}</span>
          <div className="grid grid-cols-1 gap-4">
      {graphs && (
        <>
          <img
            src={`data:image/png;base64,${graphs}`}
            alt="Graph"
          />
        </>
      )}
    </div>
        </div>
      )}
    </div>
  );
};

export default PredictForm;
