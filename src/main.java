import java.util.Scanner;
//method: use scanner to input the specific values
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        //  user to input values
        System.out.print("Enter Earned Points: ");
        double earnedPoints = scanner.nextDouble();

        System.out.print("Enter Point Total: ");
        double pointTotal = scanner.nextDouble();

        System.out.print("Enter Assignment Percentage (e.g. 35%): ");
        double assignmentPercentage = scanner.nextDouble();

        // Create an object of Score and call methods to calculate the total weighted grade
        Score totalGrade = new Score(pointTotal, earnedPoints, assignmentPercentage);
        double result = totalGrade.cal();

        // Output the result
        System.out.printf("Total weighted grade: %.5f%n", result);
        // Close scanner
        scanner.close();
    }
