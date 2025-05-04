import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:novalabs/Detials.dart';


class ShowImage extends StatefulWidget {
  final String imagePath;
  final String? userType;
  final String skin;
  Map<String, String> questions;
  ShowImage({Key? key, required this.imagePath, required this.userType,required this.skin,required this.questions}) : super(key: key);

  @override
  State<ShowImage> createState() => _ShowImageState();
}

class _ShowImageState extends State<ShowImage> {
  int indicator = 0;
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(),
      body: Center(
        child: _buildImage(),
      ),
    );
  }

  Widget _buildImage() {
    return Padding(
      padding: const EdgeInsets.all(10),
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(10),
        ),
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.only(top: 40),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(10),
                child: Image.file(
                  File(widget
                      .imagePath), // Use Image.file to display the local image
                  height: 500,
                  width: 400,
                  fit: BoxFit.cover,
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(top: 25, left: 8, right: 8),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  ElevatedButton(
                    onPressed: () async {
                      Navigator.pop(context);
                    },
                    style: ElevatedButton.styleFrom(
                      padding: EdgeInsets.symmetric(
                          horizontal: 30.0, vertical: 20.0),

                      shape: StadiumBorder(),
                    ),
                    child: Text(
                      "Retake",
                      style: TextStyle(color: Colors.white, fontSize: 18),
                    ),
                  ),
                  ElevatedButton(
                    onPressed: () async {
                      var request;
                      setState(() {
                        indicator = 1;
                      });

                      request = http.MultipartRequest('POST',
                          Uri.parse('http://192.168.74.234:8000/upload'))
                        ..files.add(await http.MultipartFile.fromPath(
                            'image', widget.imagePath));
                      String answersJson = jsonEncode(widget.questions);

                      // Add the JSON string as a field
                      request.fields['Data'] = answersJson.toString();
                      var response = await request.send();

                      if (response.statusCode == 200) {
                        // Handle success

                        Map<String, dynamic>? dataMap = (await response.stream
                            .transform(utf8.decoder)
                            .transform(json.decoder)
                            .first) as Map<String, dynamic>?;
                        setState(() {
                          indicator = 0;
                        });
                        Navigator.of(context).push(MaterialPageRoute(
                            builder: (context) => Details(
                                imagePath: widget.imagePath, result: dataMap, user1:widget.userType,skin : widget.skin,questions: widget.questions,)));
                      } else {
                        // Handle error
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            backgroundColor: Colors.greenAccent,
                            content: Text('Failed to get result, try again!'),
                          ),
                        );
                      }
                    },
                    style: ElevatedButton.styleFrom(
                      padding: EdgeInsets.symmetric(
                          horizontal: 40.0, vertical: 20.0),

                      shape: StadiumBorder(),
                    ),
                    child: Row(
                      children: [
                        Text("Confirm Image",
                            style:
                            TextStyle(color: Colors.white, fontSize: 18)),
                        SizedBox(
                          width: 5,
                        ),
                        Container(
                            height: 10,
                            width: 10,
                            child: (indicator == 1)
                                ? CircularProgressIndicator(
                              color: Colors.white,
                            )
                                : SizedBox())
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
