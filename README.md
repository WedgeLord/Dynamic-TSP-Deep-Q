# Dynamic TSP Deep Q
 
Check out the OneNote file for instructions on getting started

You'll need a /solutions folder for storing the .gif's

This model is based on the Dynamic Traveling Salesman Problem and takes an action whenever a new "request" is made. The agent makes moves inside of a loop until a new request is made. The agent gets +1 for accepting a request, 0 for declining, and -1 for failing to deliver within the request's deadline.

The environment has keywords arguments that can be used to adjust the size of the model. Keep in mind that the action space grows factorially with respect to the size of the route queue (max_cities), so be careful making the problem too large.
