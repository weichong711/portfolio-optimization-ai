import { useState } from "react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884D8"];

export default function Home() {
  const [usData, setUsData] = useState(null);
  const [myData, setMyData] = useState(null);
  const [loading, setLoading] = useState({ US: false, MY: false });

  const runPrediction = async (market) => {
    setLoading((prev) => ({ ...prev, [market]: true }));
    try {
      const res = await fetch(`${BACKEND_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ market }),
      });
      const json = await res.json();
      if (market === "US") setUsData(json);
      else setMyData(json);
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setLoading((prev) => ({ ...prev, [market]: false }));
    }
  };

  const renderPortfolio = (data) => {
    if (!data) return null;

    const pieData = data.allocation.map((item) => ({
      name: item.ticker,
      value: item.weight_pct,
    }));

    return (
      <div style={styles.card}>
        <h2 style={styles.marketTitle}>
          {data.market === "US" ? "🇺🇸 US Market" : "🇲🇾 Malaysia Market"}
        </h2>

        <div style={styles.metrics}>
          <div style={styles.metric}>
            <span style={styles.label}>Budget</span>
            <span style={styles.value}>
              {data.currency} {data.budget.toLocaleString()}
            </span>
          </div>
          <div style={styles.metric}>
            <span style={styles.label}>Expected Annual Return</span>
            <span style={{ ...styles.value, color: "#00C49F" }}>
              {data.expected_annual_return_pct}%
            </span>
          </div>
          <div style={styles.metric}>
            <span style={styles.label}>Expected Risk (Volatility)</span>
            <span style={{ ...styles.value, color: "#FF8042" }}>
              {data.expected_annual_risk_pct}%
            </span>
          </div>
          <div style={styles.metric}>
            <span style={styles.label}>Sharpe Ratio</span>
            <span style={styles.value}>{data.sharpe_ratio}</span>
          </div>
        </div>

        <h3 style={styles.sectionTitle}>Portfolio Allocation</h3>

        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={pieData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={(entry) => `${entry.name} (${entry.value}%)`}
              outerRadius={100}
              dataKey="value"
            >
              {pieData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>

        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>Stock</th>
              <th style={styles.th}>Weight</th>
              <th style={styles.th}>Amount</th>
              <th style={styles.th}>Price</th>
              <th style={styles.th}>Shares</th>
            </tr>
          </thead>
          <tbody>
            {data.allocation.map((item, i) => (
              <tr key={i} style={i % 2 === 0 ? styles.evenRow : styles.oddRow}>
                <td style={styles.td}>{item.ticker}</td>
                <td style={styles.td}>{item.weight_pct}%</td>
                <td style={styles.td}>
                  {data.currency} {item.amount.toLocaleString()}
                </td>
                <td style={styles.td}>
                  {data.currency} {item.price}
                </td>
                <td style={styles.td}>{item.shares}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>Portfolio Optimization Dashboard</h1>
        <p style={styles.subtitle}>
          Using Deep Learning (LSTM) + Swarm Intelligence (PSO)
        </p>
      </header>

      <div style={styles.buttonGroup}>
        <button
          onClick={() => runPrediction("US")}
          disabled={loading.US}
          style={styles.button}
        >
          {loading.US ? "Analyzing..." : "🇺🇸 Optimize US Portfolio ($2,500)"}
        </button>
        <button
          onClick={() => runPrediction("MY")}
          disabled={loading.MY}
          style={styles.button}
        >
          {loading.MY ? "Analyzing..." : "🇲🇾 Optimize Malaysia Portfolio (RM10,000)"}
        </button>
      </div>

      <div style={styles.results}>
        {renderPortfolio(usData)}
        {renderPortfolio(myData)}
      </div>

      <footer style={styles.footer}>
        <p>
          <strong>How it works:</strong> The system screens 40 US stocks and 25 Malaysian
          stocks, selects the top 5 by historical Sharpe ratio, predicts future returns
          using LSTM, and optimizes portfolio weights using Particle Swarm Optimization
          (PSO) to maximize risk-adjusted returns.
        </p>
      </footer>
    </div>
  );
}

const styles = {
  container: {
    fontFamily: "system-ui, -apple-system, sans-serif",
    maxWidth: "1200px",
    margin: "0 auto",
    padding: "20px",
    backgroundColor: "#f5f5f5",
    minHeight: "100vh",
  },
  header: {
    textAlign: "center",
    marginBottom: "40px",
    padding: "30px",
    backgroundColor: "white",
    borderRadius: "12px",
    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
  },
  title: {
    fontSize: "2.5rem",
    margin: "0 0 10px 0",
    color: "#333",
  },
  subtitle: {
    fontSize: "1.1rem",
    color: "#666",
    margin: 0,
  },
  buttonGroup: {
    display: "flex",
    gap: "20px",
    justifyContent: "center",
    marginBottom: "40px",
  },
  button: {
    padding: "15px 30px",
    fontSize: "1.1rem",
    fontWeight: "600",
    border: "none",
    borderRadius: "8px",
    backgroundColor: "#0070f3",
    color: "white",
    cursor: "pointer",
    boxShadow: "0 4px 12px rgba(0,112,243,0.3)",
  },
  results: {
    display: "flex",
    flexDirection: "column",
    gap: "30px",
  },
  card: {
    backgroundColor: "white",
    padding: "30px",
    borderRadius: "12px",
    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
  },
  marketTitle: {
    fontSize: "1.8rem",
    marginTop: 0,
    marginBottom: "20px",
    color: "#333",
  },
  metrics: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
    gap: "15px",
    marginBottom: "30px",
  },
  metric: {
    display: "flex",
    flexDirection: "column",
    padding: "15px",
    backgroundColor: "#f9f9f9",
    borderRadius: "8px",
  },
  label: {
    fontSize: "0.9rem",
    color: "#666",
    marginBottom: "5px",
  },
  value: {
    fontSize: "1.5rem",
    fontWeight: "700",
    color: "#333",
  },
  sectionTitle: {
    fontSize: "1.3rem",
    marginTop: "30px",
    marginBottom: "20px",
    color: "#333",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    marginTop: "20px",
  },
  th: {
    padding: "12px",
    textAlign: "left",
    backgroundColor: "#f0f0f0",
    fontWeight: "600",
    borderBottom: "2px solid #ddd",
  },
  td: {
    padding: "12px",
    borderBottom: "1px solid #eee",
  },
  evenRow: { backgroundColor: "#fafafa" },
  oddRow: { backgroundColor: "white" },
  footer: {
    marginTop: "40px",
    padding: "20px",
    backgroundColor: "white",
    borderRadius: "12px",
    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
    fontSize: "0.95rem",
    color: "#666",
    lineHeight: "1.6",
  },
};
