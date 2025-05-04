import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:novalabs/ShowIotImage.dart';
class IotDevice extends StatefulWidget {
  const IotDevice({super.key});

  @override
  State<IotDevice> createState() => _IotDeviceState();
}


class _IotDeviceState extends State<IotDevice> {
  Uint8List? imageData;
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Container(
          width: 270,
          child: OutlinedButton(

            child: Row(
              children: [

                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Text("Fetch Thermal Image",style: GoogleFonts.roboto(fontSize: 20,color: Colors.black),),
                )
              ],
              mainAxisAlignment: MainAxisAlignment.center,
            ),

            onPressed: ()async{
              var request = http.MultipartRequest('POST', Uri.parse('http://192.168.135.143:7000/fetch'));

// No need to add any files if you want to make a request without files

              var response = await request.send();

              if (response.statusCode == 200) {
                // Handle success
                List<int> bytes = await response.stream.toBytes();
                setState(() {
                  imageData = Uint8List.fromList(bytes);
                });

                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (context) => ShowIotImage(Image: imageData),
                  ),
                );
              }

            },
          ),

        ),
      )
    );
  }
}
