import "./App.css";
import { useState } from "react";
import axios from "axios";

const initialFormData = {
  duration: 0,
  src_bytes: 181,
  dst_bytes: 5450,
  land: 0,
  wrong_fragment: 0,
  urgent: 0,
  hot: 0,
  num_failed_logins: 0,
  logged_in: 1,
  num_compromised: 0,
  root_shell: 0,
  su_attempted: 0,
  num_root: 0,
  num_file_creations: 0,
  num_shells: 0,
  num_access_files: 0,
  num_outbound_cmds: 0,
  is_host_login: 0,
  is_guest_login: 0,
  count: 8,
  srv_count: 8,
  serror_rate: 0,
  srv_serror_rate: 0,
  rerror_rate: 0,
  srv_rerror_rate: 0,
  same_srv_rate: 1,
  diff_srv_rate: 0,
  srv_diff_host_rate: 0,
  dst_host_count: 9,
  dst_host_srv_count: 9,
  dst_host_same_srv_rate: 1,
  dst_host_diff_srv_rate: 0,
  dst_host_same_src_port_rate: 0.11,
  dst_host_srv_diff_host_rate: 0,
  dst_host_serror_rate: 0,
  dst_host_srv_serror_rate: 0,
  dst_host_rerror_rate: 0,
  dst_host_srv_rerror_rate: 0,
  protocol_type: "tcp",
  service: "http",
  flag: "SF"
};

function App() {
  const [formData, setFormData] = useState(initialFormData);

  const [result, setResult] = useState("");

  const [score, setScore] = useState(null);

  const [shapFeatures, setShapFeatures] = useState([]);

  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData({
      ...formData,
      [name]: value
    });
  };

  const detectAttack = async () => {
    try {
      setLoading(true);

      setResult("");
      setScore(null);
      setShapFeatures([]);

      // -------------------------------
      // Normal prediction
      // -------------------------------

      const predictionResponse = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData
      );

      setResult(predictionResponse.data.result);

      setScore(predictionResponse.data.prediction);

      // -------------------------------
      // SHAP explanation
      // -------------------------------

      const shapResponse = await axios.post(
        "http://127.0.0.1:5000/explain",
        formData
      );

      setShapFeatures(
        shapResponse.data.top_features || []
      );

    } catch (error) {
      console.error(error);

      setResult("Backend connection failed");

      setScore(null);

      setShapFeatures([]);

    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData(initialFormData);

    setResult("");

    setScore(null);

    setShapFeatures([]);
  };

  const numericalFields = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
  ];

  const services = [
    "aol",
    "auth",
    "bgp",
    "courier",
    "csnet_ns",
    "ctf",
    "daytime",
    "discard",
    "domain",
    "domain_u",
    "echo",
    "eco_i",
    "ecr_i",
    "efs",
    "exec",
    "finger",
    "ftp",
    "ftp_data",
    "gopher",
    "harvest",
    "hostnames",
    "http",
    "http_2784",
    "http_443",
    "http_8001",
    "imap4",
    "IRC",
    "iso_tsap",
    "klogin",
    "kshell",
    "ldap",
    "link",
    "login",
    "mtp",
    "name",
    "netbios_dgm",
    "netbios_ns",
    "netbios_ssn",
    "netstat",
    "nnsp",
    "nntp",
    "ntp_u",
    "other",
    "pm_dump",
    "pop_2",
    "pop_3",
    "printer",
    "private",
    "red_i",
    "remote_job",
    "rje",
    "shell",
    "smtp",
    "sql_net",
    "ssh",
    "sunrpc",
    "supdup",
    "systat",
    "telnet",
    "tftp_u",
    "tim_i",
    "time",
    "urh_i",
    "urp_i",
    "uucp",
    "uucp_path",
    "vmnet",
    "whois",
    "X11",
    "Z39_50"
  ];

  const flags = [
    "SF",
    "S0",
    "REJ",
    "RSTO",
    "RSTR",
    "SH",
    "S1",
    "S2",
    "S3",
    "RSTOS0",
    "OTH"
  ];

  const confidence =
    score !== null
      ? score <= 0.5
        ? (1 - score) * 100
        : score * 100
      : null;

  return (
    <div className="container">

      {/* ================= HEADER ================= */}

      <div className="header">

        <h1>
          Enhanced Intrusion Detection System
        </h1>

        <p>
          CNN-BiGRU-Attention with MVO, ALO and SHAP
        </p>

      </div>

      {/* ================= INPUT FORM ================= */}

      <div className="form-card">

        <h2>
          Network Traffic Input
        </h2>

        <div className="input-grid">

          {numericalFields.map((field) => (

            <div
              className="input-group"
              key={field}
            >

              <label>
                {field}
              </label>

              <input
                type="number"
                step="any"
                name={field}
                value={formData[field]}
                onChange={handleChange}
              />

            </div>

          ))}

          {/* Protocol */}

          <div className="input-group">

            <label>
              Protocol Type
            </label>

            <select
              name="protocol_type"
              value={formData.protocol_type}
              onChange={handleChange}
            >

              <option value="tcp">
                TCP
              </option>

              <option value="udp">
                UDP
              </option>

              <option value="icmp">
                ICMP
              </option>

            </select>

          </div>

          {/* Service */}

          <div className="input-group">

            <label>
              Service
            </label>

            <select
              name="service"
              value={formData.service}
              onChange={handleChange}
            >

              {services.map((service) => (

                <option
                  key={service}
                  value={service}
                >
                  {service}
                </option>

              ))}

            </select>

          </div>

          {/* Flag */}

          <div className="input-group">

            <label>
              Flag
            </label>

            <select
              name="flag"
              value={formData.flag}
              onChange={handleChange}
            >

              {flags.map((flag) => (

                <option
                  key={flag}
                  value={flag}
                >
                  {flag}
                </option>

              ))}

            </select>

          </div>

        </div>

        {/* Detect */}

        <button
          className="detect-button"
          onClick={detectAttack}
          disabled={loading}
        >

          {loading
            ? "Detecting and Explaining..."
            : "Detect Attack"}

        </button>

        {/* Reset */}

        <button
          className="reset-button"
          onClick={resetForm}
          disabled={loading}
        >

          Reset Form

        </button>

      </div>

      {/* ================= RESULT ================= */}

      {result &&
        result !== "Backend connection failed" && (

        <div className="result-card">

          <h2
            className={
              result === "Attack"
                ? "attack"
                : "normal"
            }
          >

            Prediction: {result}

          </h2>

          {confidence !== null && (

            <p>
              Confidence:{" "}
              {confidence.toFixed(2)}%
            </p>

          )}

          <p>
            Model: CNN-BiGRU-Attention
          </p>

          <p>
            Feature Selection: MVO
          </p>

          <p>
            Hyperparameter Tuning: ALO
          </p>

        </div>

      )}

      {/* ================= SHAP ================= */}

      {shapFeatures.length > 0 && (

        <div className="shap-card">

          <h2>
            SHAP Explanation
          </h2>

          <p className="shap-description">

            These features contributed most to
            the model's prediction.

          </p>

          <div className="shap-list">

            {shapFeatures.map(
              (item, index) => {

                const isPositive =
                  item.shap_value > 0;

                return (

                  <div
                    className="shap-item"
                    key={index}
                  >

                    <div className="shap-rank">

                      {index + 1}

                    </div>

                    <div className="shap-info">

                      <strong>
                        {item.feature}
                      </strong>

                      <span>
                        Feature value:{" "}
                        {item.feature_value.toFixed(4)}
                      </span>

                    </div>

                    <div
                      className={
                        isPositive
                          ? "shap-positive"
                          : "shap-negative"
                      }
                    >

                      {isPositive
                        ? "↑"
                        : "↓"}

                      {" "}

                      {Math.abs(
                        item.shap_value
                      ).toFixed(4)}

                    </div>

                  </div>

                );
              }
            )}

          </div>

          <div className="shap-note">

            <strong>
              How to read:
            </strong>

            <br />

            ↑ Positive SHAP value:
            contributes toward Attack.

            <br />

            ↓ Negative SHAP value:
            contributes toward Normal.

          </div>

        </div>

      )}

      {/* ================= ERROR ================= */}

      {result ===
        "Backend connection failed" && (

        <div className="result-card">

          <h2 className="attack">
            Backend Connection Failed
          </h2>

          <p>
            Please make sure the Flask
            backend is running.
          </p>

        </div>

      )}

    </div>
  );
}

export default App;