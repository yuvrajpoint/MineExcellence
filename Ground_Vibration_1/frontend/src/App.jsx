import React from "react";
import PredictForm from "./components/PredictForm";

const App = () => {
  return (
    <div className="bg-gray_bg min-h-screen text-white p-4">
      <h1 className="text-3xl font-bold text-orange_accent mb-6 text-center">
        Ground Vibration Predictor
      </h1>
      <PredictForm />
    </div>
  );
};

export default App;
