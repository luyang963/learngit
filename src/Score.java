public class Score {
    //initialize the attributes which is needed for calculation
        private double pointTotal;
        private double earnedPoints;
        private double assignmentPercentage;
        private double totalWeightedGrade;
        // Constructor to initialize the attributes
        public Score(double pointTotal, double earnedPoints, double assignmentPercentage) {
            this.pointTotal = pointTotal;
            this.earnedPoints = earnedPoints;
            this.assignmentPercentage = assignmentPercentage;
            this.totalWeightedGrade=0;
        }


        //method to calculate the Weighted grades
        public double cal(){
            return (earnedPoints / pointTotal) * (assignmentPercentage/100) * 100;
        }
    }

