import 'dart:convert';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
class ShowIotImage extends StatefulWidget {
  final Uint8List? Image;
  const ShowIotImage({super.key,required this.Image});

  @override
  State<ShowIotImage> createState() => _ShowIotImageState();
}

class _ShowIotImageState extends State<ShowIotImage> {
  List<CameraDescription>? cameras;
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
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
                  child: Image.memory(
                    widget.Image!,
                  )
                ),
              ),
              Padding(
                padding: const EdgeInsets.only(top: 25, left: 8, right: 8),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    ElevatedButton(
                      onPressed: () async {
                        var request;


                        request = http.MultipartRequest('POST',
                            Uri.parse('http://192.168.161.64:7000/temperature'))
                          ..files.add(await http.MultipartFile.fromBytes(
                              'image', widget.Image!,filename: 'Image.jpg'));

                        var response = await request.send();

                        if (response.statusCode == 200) {
                          // Handle success

                          Map<String, dynamic>? dataMap = (await response.stream
                              .transform(utf8.decoder)
                              .transform(json.decoder)
                              .first) as Map<String, dynamic>?;
                          print(dataMap);
                          // Navigator.of(context).push(MaterialPageRoute(builder: (context)=>TakePictureScreen2(cameras: cameras,data : dataMap!['difference'])));
                          // Navigator.of(context).push(MaterialPageRoute(
                          //     builder: (context) => Details(
                          //       imagePath: widget.imagePath, result: dataMap, user1:widget.userType,skin : widget.skin,questions: widget.questions,)));
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
                            horizontal: 30.0, vertical: 20.0),

                        shape: StadiumBorder(),
                      ),
                      child: Text(
                        "Confirm",
                        style: TextStyle(color: Colors.white, fontSize: 18),
                      ),
                    ),

                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
