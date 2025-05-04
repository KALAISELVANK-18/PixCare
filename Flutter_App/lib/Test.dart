import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class RoundedContainerStack extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Stack(
      children: <Widget>[
        // Bottom Container with Rounded Corners
        Positioned(
          top: 0,
          left: 0,
          right: 0,
          child: Container(
            height: 200, // Adjust the height as needed
            decoration: BoxDecoration(
              color: Colors.blue,
              borderRadius: BorderRadius.only(
                topLeft: Radius.circular(20.0),
                topRight: Radius.circular(20.0),
              ),
            ),
          ),
        ),
        // Top Container (you can add content here)
        Positioned(
          top: 20, // Adjust the position as needed
          left: 20, // Adjust the position as needed
          right: 20, // Adjust the position as needed
          child: Container(
            height: 160, // Adjust the height as needed
            decoration: BoxDecoration(
              color: Colors.white,
              boxShadow: [
                BoxShadow(
                  color: Colors.grey.withOpacity(0.5),
                  spreadRadius: 2,
                  blurRadius: 7,
                  offset: Offset(0, 3),
                ),
              ],
            ),
            // You can add your content here
            child: Center(
              child: Text(
                'Top Container Content',
                style: TextStyle(fontSize: 18),
              ),
            ),
          ),
        ),
      ],
    );
  }
}
