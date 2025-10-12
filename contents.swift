import SwiftUI
import PlaygroundSupport

struct CalculatorView: View {
    @State private var firstInput = ""
    @State private var secondInput = ""
    @State private var selectedOperation: Operation = .addition

    private enum Operation: String, CaseIterable, Identifiable {
        case addition = "+"
        case subtraction = "-"
        case multiplication = "*"
        case division = "/"

        var id: String { rawValue }

        func perform(_ lhs: Double, _ rhs: Double) -> Double? {
            switch self {
            case .addition:
                return lhs + rhs
            case .subtraction:
                return lhs - rhs
            case .multiplication:
                return lhs * rhs
            case .division:
                return rhs == 0 ? nil : lhs / rhs
            }
        }
    }

    private var firstValue: Double? {
        Double(firstInput.replacingOccurrences(of: ",", with: "."))
    }

    private var secondValue: Double? {
        Double(secondInput.replacingOccurrences(of: ",", with: "."))
    }

    private var resultText: String {
        guard let lhs = firstValue, let rhs = secondValue else {
            return "Enter two numbers"
        }
        guard let result = selectedOperation.perform(lhs, rhs) else {
            return "Cannot divide by zero"
        }

        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.maximumFractionDigits = 6

        return formatter.string(from: NSNumber(value: result)) ?? "\(result)"
    }

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Values")) {
                    TextField("First number", text: $firstInput)
                        .keyboardType(.decimalPad)
                    TextField("Second number", text: $secondInput)
                        .keyboardType(.decimalPad)
                }

                Section(header: Text("Operation")) {
                    Picker("Operation", selection: $selectedOperation) {
                        ForEach(Operation.allCases) { operation in
                            Text(operation.rawValue).tag(operation)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                Section(header: Text("Result")) {
                    Text(resultText)
                        .font(.title3)
                        .bold()
                        .foregroundColor(resultText == "Cannot divide by zero" ? .red : .primary)
                }
            }
            .navigationTitle("Calculator")
        }
    }
}

PlaygroundPage.current.setLiveView(CalculatorView())
