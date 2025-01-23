public class HelloWorld {
    //create attribute to store the greeting message
        private String greeting = "Hello, World!";
    //initialize attribute through setter and getter
        public void setGreeting(String greeting){
        this.greeting = greeting;
        }
        public  String getGreeting(){
            return  greeting;
        }
    //build method to print the value of attribute;
        public void greet(){
        System.out.println(greeting);
        }
    
    // Main method is the entry point of the program
        public static void main (String args[]){
        // Create an instance of the HelloWorld class
        HelloWorld n = new HelloWorld ();
        //  Use the setGreeting method to change the greeting message
        n.setGreeting("Hello, java!");
        // Call the greet method to print the updated greeting message
        n.greet();
        } 
    }
    