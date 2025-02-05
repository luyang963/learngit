
// Main method to run the program
public static void main() {
    //create object
    WeightedGrades weightedGrades = new WeightedGrades();

    // Step 1: call out the relative method to input grades and percentages
    weightedGrades.inputGradesAndPercentages();

    // Step 2: call method to calculate the weighted grade
    weightedGrades.calculateWeightedGrade();

    // Step 3: call method to display the results
    weightedGrades.displayResults();
}
