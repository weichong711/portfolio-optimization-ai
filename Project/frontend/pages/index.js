import { useState } from "react";

export default function Home() {
  const [data, setData] = useState(null);

  const runPrediction = async (market) => {
    const res = await fetch("https://your-backend-url/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ market })
    });

    const json = await res.json();
    setData(json);
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Portfolio Optimization Dashboard</h1>

      <button onClick={() => runPrediction("US")}>US Market</button>
      <button onClick={() => runPrediction("MY")}>Malaysia Market</button>

      {data && (
        <div>
          <h2>Optimized Weights</h2>
          {data.stocks.map((s, i) => (
            <p key={i}>
              {s}: {(data.weights[i] * 100).toFixed(2)}%
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
