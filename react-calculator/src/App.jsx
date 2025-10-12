import { useMemo, useState } from 'react';
import './App.css';

const OPERATIONS = [
  { label: '+', value: 'add', handler: (a, b) => a + b },
  { label: '-', value: 'subtract', handler: (a, b) => a - b },
  { label: '*', value: 'multiply', handler: (a, b) => a * b },
  { label: '/', value: 'divide', handler: (a, b) => (b === 0 ? null : a / b) },
];

const formatter = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 6,
});

const parseNumber = (value) =>
  value.trim() === '' ? null : Number(value.replace(',', '.'));

function App() {
  const [firstInput, setFirstInput] = useState('');
  const [secondInput, setSecondInput] = useState('');
  const [selectedOperation, setSelectedOperation] = useState(OPERATIONS[0]);

  const result = useMemo(() => {
    const first = parseNumber(firstInput);
    const second = parseNumber(secondInput);

    if (Number.isNaN(first) || Number.isNaN(second)) {
      return { message: 'Enter valid numbers', isError: true };
    }

    if (first === null || second === null) {
      return { message: 'Enter two numbers', isError: false };
    }

    const raw = selectedOperation.handler(first, second);

    if (raw === null) {
      return { message: 'Cannot divide by zero', isError: true };
    }

    return {
      message: formatter.format(raw),
      isError: false,
    };
  }, [firstInput, secondInput, selectedOperation]);

  return (
    <div className="app">
      <h1 className="title">Calculator</h1>

      <div className="card">
        <label className="input-group">
          <span>First number</span>
          <input
            aria-label="First number"
            inputMode="decimal"
            value={firstInput}
            onChange={(event) => setFirstInput(event.target.value)}
          />
        </label>

        <label className="input-group">
          <span>Second number</span>
          <input
            aria-label="Second number"
            inputMode="decimal"
            value={secondInput}
            onChange={(event) => setSecondInput(event.target.value)}
          />
        </label>

        <div className="operations">
          {OPERATIONS.map((operation) => (
            <button
              key={operation.value}
              type="button"
              className={
                operation.value === selectedOperation.value
                  ? 'operation-button operation-button--active'
                  : 'operation-button'
              }
              onClick={() => setSelectedOperation(operation)}
            >
              {operation.label}
            </button>
          ))}
        </div>

        <div
          role="status"
          aria-live="polite"
          className={`result ${result.isError ? 'result--error' : ''}`}
        >
          {result.message}
        </div>
      </div>
    </div>
  );
}

export default App;
