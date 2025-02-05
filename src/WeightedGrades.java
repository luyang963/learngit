import java.util.Scanner;

    public class WeightedGrades {

        // Initialize attributes to store grades and their percentages
        private double[] grades;       // Array to store 8 grades
        private double[] percentages;  // Array to store 8 percentages
        private double totalWeightedGrade; // Final weighted grade

        // Constructor to initialize arrays
        public WeightedGrades() {
            grades = new double[8];
            percentages = new double[8];
        }

        // Method to input grades and percentages from the user
        public void inputGradesAndPercentages() {
            //scanner standard Library Utils for user input
            Scanner scanner = new Scanner(System.in);
            //input the grades and store to the array
            System.out.println("Enter the 8 grades :");
            for (int i = 0; i < 8; i++) {

                System.out.print("Grade " + (i + 1) + ": ");
                grades[i] = scanner.nextDouble();
            }
           //input the percentages and store to the array
            System.out.println("Enter the 8 percentages :");
            for (int i = 0; i < 8; i++) {
                System.out.print("Percentage " + (i + 1) + ": ");
                percentages[i] = scanner.nextDouble();
            }
        }

        // Method to calculate the total weighted grade
        public void calculateWeightedGrade() {
            totalWeightedGrade = 0;
            for (int i = 0; i < 8; i++) {
                //use sum
                totalWeightedGrade += grades[i] * (percentages[i] / 100);
            }
        }

        //  if-then logic, Method to determine the letter grade
        public String determineLetterGrade() {
                if (totalWeightedGrade >= 90 && totalWeightedGrade <= 100) {
                    return "A";
                } else if (totalWeightedGrade >= 80 && totalWeightedGrade < 90) {
                    return "B";
                } else if (totalWeightedGrade >= 70 && totalWeightedGrade <80) {
                    return "C";
                } else if (totalWeightedGrade >= 60 && totalWeightedGrade < 70) {
                    return "D";
                } else {
                    return "F";
            }
        }

        // Method to display the results
        public void displayResults() {
            System.out.printf("Total Weighted Grade: %.5f%n", totalWeightedGrade);
            System.out.println("Letter Grade: " + determineLetterGrade());
        }

    }
