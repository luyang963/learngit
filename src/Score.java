public class Score {
    //initialize the attributes which is needed for calculation
        // Attributes to hold values for calculation
        private double pointTotal;       // Total points possible (
        private double earnedPoints;     // Points earned by the student
        private double assignmentPercentage; // Weight of the assignment
        private double totalWeightedGrade; // Final weighted grade

        // Constructor to initialize the attributes
        public Score(double pointTotal, double earnedPoints, double assignmentPercentage) {
            this.pointTotal = pointTotal;
            this.earnedPoints = earnedPoints;
            this.assignmentPercentage = assignmentPercentage;
            this.totalWeightedGrade=0;
        }

        //method to calculate the Weighted
        public double cal(){
            return (earnedPoints / pointTotal) * (assignmentPercentage/100) * 100;
        }
    }

