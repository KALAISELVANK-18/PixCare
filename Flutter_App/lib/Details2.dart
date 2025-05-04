import 'dart:io';

import 'package:carousel_slider/carousel_slider.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
// import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:dropdown_search/dropdown_search.dart';
import 'package:novalabs/Maps.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:shared_preferences/shared_preferences.dart';


class Details2 extends StatefulWidget {
  final String imagePath;
  final List<DocumentSnapshot<Object?>> result;
  int index;
  final String? usertype;

  Details2(
      {super.key,
      required this.imagePath,
      required this.result,
      required this.index,
        required this.usertype,
      });

  @override
  State<Details2> createState() => _Details2State();
}

class _Details2State extends State<Details2> {
  int post = 0;
  final user = FirebaseAuth.instance.currentUser!.email;
  final List<String> sliderItems = [
    "list.png",
    "3dmap.jpg",
  ];
  String? language;
  void initState() {

    getLanguage();
  }
  Future<String?> getLanguage() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    setState(() {
      language = prefs.getString("userLanguage");
    });
    return null;
  }


  String selectedDisease = "Acne and Rosacea";


  final List<String> Diseases = [
    "Acne and Rosacea","Actinic Keratosis Basal Cell Carcinoma",
    "Atopic Dermatitis","Bullous Disease","Cellulitis Impetigo and other Bacterial","Eczema","Exanthems and Drug Eruptions",
    "Hair Loss,Alopecia and others","Herpes HPV and other STDs","Light Diseases and Disorders of Pigmentation","Lupus and other connective Tissue",
    "Melanoma Skin Cancer","Nail Fungus and other Nail diseases","Poison Ivy","Psoriasis,Lichen Planus and related","Scabies Lyme and other Infestations",
    "Seberrheic keratoses and other benign tumors","Systemic Disease","Tinea Ringworm Candidiasis","Urticaria Hives","Vascular Tumors","Vasculitis","Warts molluscum and other viral Infections"
  ];
  sendmeassage(String token,String title,String imageurl,String body)async{

    try{
      await http.post(
        Uri.parse('https://fcm.googleapis.com/fcm/send'),
        headers: <String,String>{
          'Content-Type':'application/json',
          'Authorization':'key=AAAA5q759rw:APA91bHb7cEygJCcCtYmc3FJxlZBnsYY1UN7ilSMqRRbauqDP8UBdAAY2Xn19BF4U-tv8A2BV1JUjFLtyuB_bfd8PXtdcBa0mVpfzMqzFiQ3yMNBwmJjYtSj6OAS914hycPz5vK6gRi4'
        },
        body: jsonEncode(
          <String,dynamic>{
            'priority':'high',
            'data':<String,dynamic>{
              'click_action':'FLUTTER_NOTIFICATION_CLICK',
              'status':'done',
              'body':body,
              'title':title,
              "image":imageurl,

            },
            "notification":<String,dynamic>{
              "title":title,
              "body":body,
              "image":imageurl,
              "android_channel_id":"dbnoti"
            },
            "to":token,
          },
        ),
      );
    }

    catch(e){
      if(kDebugMode){
        print("error push notification");

      }
    }
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          image: DecorationImage(
            image: AssetImage("images/bg.png"),
            fit: BoxFit.cover,
          ),
        ),
        child: SingleChildScrollView(
          child: Column(
            children: [
              const Padding(
                padding: EdgeInsets.only(top: 40),
                child: Row(
                  children: [
                    Padding(
                      padding: EdgeInsets.symmetric(
                          horizontal: 20, vertical: 20),
                      child: Text(
                        "Report ",
                        style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.w600,
                            fontSize: 30),
                      ),
                    ),
                  ],
                ),
              ),
              Padding(
                padding: const EdgeInsets.only(top: 20),
                child: Container(
                  height: 1100,
                  width: MediaQuery.of(context).size.width * 1,
                  decoration: BoxDecoration(
                    color: Color.fromRGBO(229, 235, 255, 1),
                    borderRadius: BorderRadius.only(
                      topLeft: Radius.circular(30),
                      topRight: Radius.circular(30),
                    ),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(15),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        CarouselSlider(
                          options: CarouselOptions(
                            height: 300.0,
                            enlargeCenterPage: true,
                            aspectRatio: 16 / 9,
                            enableInfiniteScroll: true,
                            autoPlayCurve: Curves.fastOutSlowIn,
                            viewportFraction: 0.8,
                          ),
                          items: sliderItems.map((item) {
                            return Builder(
                              builder: (BuildContext context) {
                                return Container(
                                  width: MediaQuery.of(context).size.width,
                                  margin: EdgeInsets.symmetric(
                                      horizontal: 5.0, vertical: 10),
                                  decoration: BoxDecoration(
                                    image: DecorationImage(
                                      image: NetworkImage(
                                        widget.imagePath,
                                      ),
                                      fit: BoxFit.cover,
                                    ),
                                    color: Colors.white,
                                    borderRadius: BorderRadius.circular(10),
                                  ),
                                  child: Center(),
                                );
                              },
                            );
                          }).toList(),
                        ),
                        (widget.result[widget.index]["diseases"])
                            ? Padding(
                                padding: const EdgeInsets.all(1),
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Padding(
                                      padding: const EdgeInsets.all(8.0),
                                      child: Row(
                                        children: [
                                          Text(
                                            "Result*:  ",
                                            style: TextStyle(
                                                fontSize: 18,
                                                fontWeight: FontWeight.w600),
                                          ),
                                          Text(
                                            (double.parse(widget.result[widget.index]
                                                            ["accuracy1"]) *
                                                        100)
                                                    .round()
                                                    .toString() +
                                                "% Pathology Found",
                                            style: TextStyle(
                                                fontSize: 18,
                                                fontWeight: FontWeight.w400),
                                          ),
                                        ],
                                      ),
                                    ),
                                    // Padding(
                                    //   padding: const EdgeInsets.all(10),
                                    //   child: SizedBox(
                                    //     height:
                                    //         300, // Adjust the height of the bar chart as needed
                                    //     child: BarChart(
                                    //       BarChartData(
                                    //         alignment:
                                    //             BarChartAlignment.spaceAround,
                                    //         maxY:
                                    //             100, // Set the maximum value (100%)
                                    //         barGroups: [
                                    //           BarChartGroupData(
                                    //             x: 0,
                                    //             barRods: [
                                    //               BarChartRodData(
                                    //                   y: double.parse(widget.result[
                                    //                               widget.index]
                                    //                           ["accuracy1"]) *
                                    //                       100,
                                    //                   colors: [Colors.blue]),
                                    //             ],
                                    //           ),
                                    //           BarChartGroupData(
                                    //             x: 1,
                                    //             barRods: [
                                    //               BarChartRodData(
                                    //                   y: double.parse(widget.result[
                                    //                               widget.index]
                                    //                           ["accuracy2"]) *
                                    //                       100,
                                    //                   colors: [Colors.green]),
                                    //             ],
                                    //           ),
                                    //           BarChartGroupData(
                                    //             x: 2,
                                    //             barRods: [
                                    //               BarChartRodData(
                                    //                   y: double.parse(widget.result[
                                    //                               widget.index]
                                    //                           ["accuracy3"]) *
                                    //                       100,
                                    //                   colors: [Colors.orange]),
                                    //             ],
                                    //           ),
                                    //         ],
                                    //         titlesData: FlTitlesData(
                                    //           leftTitles:
                                    //               SideTitles(showTitles: true),
                                    //           bottomTitles:
                                    //               SideTitles(showTitles: true),
                                    //         ),
                                    //       ),
                                    //     ),
                                    //   ),
                                    // ),
                                    Padding(
                                      padding: const EdgeInsets.all(8.0),
                                      child: Row(
                                        children: [
                                          Container(
                                            child: Text(
                                              "Diagnosis:  ",
                                              style: TextStyle(
                                                  fontSize: 18,
                                                  fontWeight: FontWeight.w600),
                                            ),
                                          ),
                                          Container(
                                            width: 230,
                                            child: Text(
                                              widget.result[widget.index]["title"],
                                              softWrap: true,
                                              style: TextStyle(
                                                  fontSize: 18,
                                                  fontWeight: FontWeight.w400),
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                    Padding(
                                      padding: const EdgeInsets.all(8.0),
                                      child: Row(
                                        children: [
                                          Text(
                                            "Advice:  ",
                                            style: TextStyle(
                                                fontSize: 18,
                                                fontWeight: FontWeight.w600),
                                          ),
                                          Container(
                                            width: 250,
                                            child: Text(
                                              "Take visit to dermatologist",
                                              softWrap: true,
                                              style: TextStyle(
                                                  fontSize: 18,
                                                  fontWeight: FontWeight.w400),
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                    Padding(
                                      padding: const EdgeInsets.all(8.0),
                                      child: StreamBuilder<QuerySnapshot>(

                                          stream: FirebaseFirestore.instance.collection('Precautions').where('name',isEqualTo: 'Diseases and Precautions').snapshots(),
                                          builder: (context, snapshot) {
                                            if (snapshot.connectionState ==
                                                ConnectionState.waiting) {
                                              return CircularProgressIndicator();
                                            }

                                            if (snapshot.hasError) {
                                              return Text('Error: ${snapshot.error}');
                                            }

                                            if (!snapshot.hasData ||
                                                snapshot.data!.docs.isEmpty) {
                                              return Text('No data available');
                                            }

                                            // Get the document for the current index
                                            var document = snapshot.data!.docs[0];
                                            var data = document.data() as Map<
                                                String,
                                                dynamic>;

                                            String res = widget.result[widget.index]["title"].toString().substring(0,widget.result[widget.index]["title"].toString().length-1);

                                            return Column(
                                              crossAxisAlignment: CrossAxisAlignment.start,
                                              children: [
                                                Text(
                                                  language == 'hindi' ? "सलाह:" : "Precautions:  ",
                                                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600),
                                                ),
                                                // Use a ListView.builder to dynamically build the list of precautions
                                                ListView.builder(
                                                  shrinkWrap: true,
                                                  itemCount: data[res].length,
                                                  itemBuilder: (context, index) {
                                                    // Split the string into the English and Hindi parts
                                                    List<String> parts = data[res][index].split("/");
                                                    String englishText = parts[0];
                                                    String hindiText = parts.length > 1 ? parts[1] : "";

                                                    return Padding(
                                                      padding: const EdgeInsets.only(top: 0),
                                                      child: Container(
                                                        width: 450,
                                                        child: Text(
                                                          language == 'hindi' ? hindiText : englishText,
                                                          softWrap: true,
                                                          style: TextStyle(fontSize: 18, fontWeight: FontWeight.w400),
                                                        ),
                                                      ),
                                                    );
                                                  },
                                                ),
                                              ],
                                            );
                                          }
                                      ),
                                    ),
                                    Padding(
                                      padding: const EdgeInsets.symmetric(
                                          vertical: 20, horizontal: 60),
                                      child: Row(
                                        children: [
                                          OutlinedButton.icon(
                                            onPressed: () {
                                              Navigator.of(context).push(
                                                MaterialPageRoute(
                                                  builder: (context) =>
                                                      Myapps(),
                                                ),
                                              );
                                            },
                                            icon: Icon(Icons.location_city),
                                            label: Text(
                                              "Find Dermatologist",
                                              style: TextStyle(fontSize: 18),
                                            ),
                                          )
                                        ],
                                      ),
                                    ),
                                    Padding(
                                      padding: const EdgeInsets.all(8.0),
                                      child: Container(
                                        width: 500,
                                        child: Text(
                                          "* This scan result is not a diagnosis. Please consult your doctor for an accurate diagnosis and treatment",
                                          softWrap: true,
                                          style: TextStyle(
                                              fontSize:
                                                  16), // Enable text wrapping
                                        ),
                                      ),
                                    ),
                                    (widget.usertype=='Doctor')?
                                    Padding(
                                      padding: EdgeInsets.only(top: 10,left: 5,right: 5),

                                        child: Column(
                                          children: [
                                            Text(
                                              "If you are not satisfied with the result",
                                              softWrap: true,
                                              style: TextStyle(
                                                  fontSize:
                                                  16), // Enable text wra
                                            ),
                                            TextButton(onPressed: ()=>showDialog<String>(

                                                context: context,

                                                builder: (BuildContext context) {
                                                  TextEditingController disease=new TextEditingController();
                                                  TextEditingController reason=new TextEditingController();


                                                  return SingleChildScrollView(
                                                    child: AlertDialog(

                                                      title: Column(
                                                        children: [

                                                          Center(child: Text("Your Assumption")),
                                                          Padding(
                                                              padding: EdgeInsets.only(top: 20, left: 20, right: 20),
                                                              child: DropdownSearch<String>(
                                                    popupProps: PopupProps.menu(
                                                    showSelectedItems: true,
                                                    showSearchBox: true,

                                                    ),
                                                    items: Diseases,
                                                    dropdownDecoratorProps: DropDownDecoratorProps(
                                                    dropdownSearchDecoration: InputDecoration(
                                                    labelText: "Disease",
                                                    hintText: "Enter your assumption",

                                                    border: OutlineInputBorder(
                                                      borderRadius: BorderRadius.circular(10),
                                                    )

                                                    ),
                                                    ),
                                                    onChanged: (newValue){
                                                      setState((){
                                                        selectedDisease = newValue!;
                                                      });
                                                    },
                                                    selectedItem: selectedDisease,
                                                    ),
                                                    ),
                                                          Padding(
                                                              padding: EdgeInsets.only(top: 20, left: 20, right: 20),
                                                              child: TextFormField(

                                                                validator: (value) {
                                                                  if (value!.isEmpty) {
                                                                    return "Please enter password";
                                                                  }
                                                                  return null;
                                                                },
                                                                controller: reason,
                                                                decoration: InputDecoration(
                                                                  filled: true, //<-- SEE HERE
                                                                  fillColor: Colors.white,
                                                                  border: OutlineInputBorder(
                                                                      borderRadius: BorderRadius.circular(10)),
                                                                  hintText: 'Enter your reason',
                                                                ),

                                                              )),

                                                          SizedBox(height: 10,),
                                                          Padding(
                                                            padding: const EdgeInsets.all(20.0),
                                                            child: Row(
                                                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                                              children: <Widget>[
                                                               OutlinedButton(
                                                                   onPressed: (){
                                                                     Navigator.pop(context,"ok");
                                                                   },
                                                                   child: Text("Cancel")
                                                               ),
                                                                OutlinedButton(
                                                                    onPressed: ()async{
                                                                      final c = DateTime.timestamp();

                                                                      final refe = FirebaseStorage
                                                                          .instance
                                                                          .ref()
                                                                          .child(FirebaseAuth.instance
                                                                          .currentUser!.uid)
                                                                          .child(c.toString())
                                                                          .child("Reinforcement");

                                                                      await refe.putFile(
                                                                          File(widget.imagePath),
                                                                          SettableMetadata(
                                                                              contentType:
                                                                              "image/png"));

                                                                      final downloadUrl1 =
                                                                      await FirebaseStorage
                                                                          .instance
                                                                          .ref()
                                                                          .child(FirebaseAuth
                                                                          .instance
                                                                          .currentUser!
                                                                          .uid)
                                                                          .child(c.toString())
                                                                          .child("Reinforcement")
                                                                          .getDownloadURL();
                                                                      final String url1 =
                                                                      downloadUrl1.toString();

                                                                      await FirebaseFirestore.instance.collection("Reinforcement").doc().set({
                                                                        "Disease":selectedDisease,
                                                                        "Reason": reason.text.toString(),
                                                                        "Yes":0,
                                                                        "No":0,
                                                                        "Active":true,
                                                                        "Voters":[],
                                                                        "user":FirebaseAuth.instance.currentUser!.email.toString(),
                                                                        "predictedDisease":"No Disease",
                                                                        "imageUrl":url1,
                                                                        "time": DateTime
                                                                            .timestamp(),
                                                                      });
                                                                      Notification(url1,selectedDisease,reason.text.toString());
                                                                      Navigator.pop(context,"ok");
                                                                    },

                                                                    child: Text("Submit")
                                                                ),
                                                              ],
                                                            ),
                                                          )
                                                        ],
                                                      ),


                                                    ),
                                                  );}
                                            ),


                                            child: Text("Click here!",style: TextStyle(fontSize: 16,),textAlign: TextAlign.center,)
                                            ),
                                          ],
                                        ),



                                    ):
                                        SizedBox()
                                  ],
                                ),
                              )
                            : Padding(
                                padding: const EdgeInsets.all(1),
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Padding(
                                      padding: const EdgeInsets.all(8.0),
                                      child: Row(
                                        children: [
                                          Text(
                                            "Result*:  ",
                                            style: TextStyle(
                                                fontSize: 18,
                                                fontWeight: FontWeight.w600),
                                          ),
                                          Text(
                                            "No Pathology Found",
                                            style: TextStyle(
                                                fontSize: 18,
                                                fontWeight: FontWeight.w400),
                                          ),
                                        ],
                                      ),
                                    ),
                                    // Padding(
                                    //   padding: const EdgeInsets.all(10),
                                    //   child: SizedBox(
                                    //     height:
                                    //     300, // Adjust the height of the bar chart as needed
                                    //     child: BarChart(
                                    //       BarChartData(
                                    //         alignment: BarChartAlignment.spaceAround,
                                    //         maxY: 100, // Set the maximum value (100%)
                                    //         barGroups: [
                                    //           BarChartGroupData(
                                    //             x: 0,
                                    //             barRods: [
                                    //               BarChartRodData(
                                    //
                                    //                   y: result!["result"][0][1].toInt(), colors: [Colors.blue]),
                                    //             ],
                                    //           ),
                                    //           BarChartGroupData(
                                    //             x: 1,
                                    //             barRods: [
                                    //               BarChartRodData(
                                    //                   y: result!["result"][1][1].toInt(), colors: [Colors.green]),
                                    //             ],
                                    //           ),
                                    //           BarChartGroupData(
                                    //             x: 2,
                                    //             barRods: [
                                    //               BarChartRodData(
                                    //                   y: result!["result"][2][1].toInt(), colors: [Colors.orange]),
                                    //             ],
                                    //           ),
                                    //         ],
                                    //         titlesData: FlTitlesData(
                                    //           leftTitles:
                                    //           SideTitles(showTitles: true),
                                    //           bottomTitles:
                                    //           SideTitles(showTitles: true),
                                    //         ),
                                    //       ),
                                    //     ),
                                    //   ),
                                    // ),
                                    Padding(
                                      padding: const EdgeInsets.all(8.0),
                                      child: Row(
                                        children: [
                                          Text(
                                            "Diagnosis:  ",
                                            style: TextStyle(
                                                fontSize: 18,
                                                fontWeight: FontWeight.w600),
                                          ),
                                          Text(
                                            "No Disease Found",
                                            style: TextStyle(
                                                fontSize: 18,
                                                fontWeight: FontWeight.w400),
                                          ),
                                        ],
                                      ),
                                    ),
                                    Padding(
                                      padding: const EdgeInsets.all(8.0),
                                      child: Row(
                                        children: [
                                          Text(
                                            "Advice:  ",
                                            style: TextStyle(
                                                fontSize: 18,
                                                fontWeight: FontWeight.w600),
                                          ),
                                          Container(
                                            width: 250,
                                            child: Text(
                                              "No risk! Advisable to visit Doctor",
                                              softWrap: true,
                                              style: TextStyle(
                                                  fontSize: 18,
                                                  fontWeight: FontWeight.w400),
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                    Padding(
                                      padding: const EdgeInsets.symmetric(
                                          vertical: 20, horizontal: 60),
                                      child: Row(
                                        children: [
                                          OutlinedButton.icon(
                                            onPressed: () {
                                              Navigator.of(context).push(
                                                MaterialPageRoute(
                                                  builder: (context) =>
                                                      Myapps(),
                                                ),
                                              );
                                            },
                                            icon: Icon(Icons.location_city),
                                            label: Text(
                                              "Find Dermatologist",
                                              style: TextStyle(fontSize: 18),
                                            ),
                                          )
                                        ],
                                      ),
                                    ),
                                    Padding(
                                      padding: const EdgeInsets.all(8.0),
                                      child: Container(
                                        width: 500,
                                        child: Text(
                                          "* This scan result is not a diagnosis. Please consult your doctor for an accurate diagnosis and treatment",
                                          softWrap: true,
                                          style: TextStyle(
                                              fontSize:
                                                  16), // Enable text wrapping
                                        ),
                                      ),
                                    ),
                                    (widget.usertype=='Doctor')?
                                    Padding(
                                      padding: EdgeInsets.only(top: 10,left: 5,right: 5),

                                      child: Column(
                                        children: [
                                          Text(
                                            "If you are not satisfied with the result",
                                            softWrap: true,
                                            style: TextStyle(
                                                fontSize:
                                                16), // Enable text wra
                                          ),
                                          TextButton(onPressed: ()=>showDialog<String>(

                                              context: context,

                                              builder: (BuildContext context) {
                                                TextEditingController disease=new TextEditingController();
                                                TextEditingController reason=new TextEditingController();


                                                return SingleChildScrollView(
                                                  child: AlertDialog(

                                                    title: Column(
                                                      children: [

                                                        Center(child: Text("Your Assumption")),
                                                        Padding(
                                                          padding: EdgeInsets.only(top: 20, left: 20, right: 20),
                                                          child: DropdownSearch<String>(
                                                            popupProps: PopupProps.menu(
                                                              showSelectedItems: true,
                                                              showSearchBox: true,

                                                            ),
                                                            items: Diseases,
                                                            dropdownDecoratorProps: DropDownDecoratorProps(
                                                              dropdownSearchDecoration: InputDecoration(
                                                                  labelText: "Disease",
                                                                  hintText: "Enter your assumption",

                                                                  border: OutlineInputBorder(
                                                                    borderRadius: BorderRadius.circular(10),
                                                                  )

                                                              ),
                                                            ),
                                                            onChanged: (newValue){
                                                              setState((){
                                                                selectedDisease = newValue!;
                                                              });
                                                            },
                                                            selectedItem: selectedDisease,
                                                          ),
                                                        ),
                                                        Padding(
                                                            padding: EdgeInsets.only(top: 20, left: 20, right: 20),
                                                            child: TextFormField(

                                                              validator: (value) {
                                                                if (value!.isEmpty) {
                                                                  return "Please enter password";
                                                                }
                                                                return null;
                                                              },
                                                              controller: reason,
                                                              decoration: InputDecoration(
                                                                filled: true, //<-- SEE HERE
                                                                fillColor: Colors.white,
                                                                border: OutlineInputBorder(
                                                                    borderRadius: BorderRadius.circular(10)),
                                                                hintText: 'Enter your reason',
                                                              ),

                                                            )),

                                                        SizedBox(height: 10,),
                                                        Padding(
                                                          padding: const EdgeInsets.all(20.0),
                                                          child: Row(
                                                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                                            children: <Widget>[
                                                              OutlinedButton(
                                                                  onPressed: (){
                                                                    Navigator.pop(context,"ok");
                                                                  },
                                                                  child: Text("Cancel")
                                                              ),
                                                              OutlinedButton(
                                                                  onPressed: ()async{
                                                                    final c = DateTime.timestamp();

                                                                    final refe = FirebaseStorage
                                                                        .instance
                                                                        .ref()
                                                                        .child(FirebaseAuth.instance
                                                                        .currentUser!.uid)
                                                                        .child(c.toString())
                                                                        .child("Reinforcement");

                                                                    await refe.putFile(
                                                                        File(widget.imagePath),
                                                                        SettableMetadata(
                                                                            contentType:
                                                                            "image/png"));

                                                                    final downloadUrl1 =
                                                                    await FirebaseStorage
                                                                        .instance
                                                                        .ref()
                                                                        .child(FirebaseAuth
                                                                        .instance
                                                                        .currentUser!
                                                                        .uid)
                                                                        .child(c.toString())
                                                                        .child("Reinforcement")
                                                                        .getDownloadURL();
                                                                    final String url1 =
                                                                    downloadUrl1.toString();

                                                                    await FirebaseFirestore.instance.collection("Reinforcement").doc().set({
                                                                      "Disease":selectedDisease,
                                                                      "Reason": reason.text.toString(),
                                                                      "Yes":0,
                                                                      "No":0,
                                                                      "Active":true,
                                                                      "Voters":[],
                                                                      "user":FirebaseAuth.instance.currentUser!.email.toString(),
                                                                      "predictedDisease":"No Disease",
                                                                      "imageUrl":url1,
                                                                      "time": DateTime
                                                                          .timestamp(),
                                                                    });
                                                                    Notification(url1,selectedDisease,reason.text.toString());
                                                                    Navigator.pop(context,"ok");
                                                                  },

                                                                  child: Text("Submit")
                                                              ),
                                                            ],
                                                          ),
                                                        )
                                                      ],
                                                    ),


                                                  ),
                                                );}
                                          ),


                                              child: Text("Click here!",style: TextStyle(fontSize: 16,),textAlign: TextAlign.center,)
                                          ),
                                        ],
                                      ),



                                    ):
                                        SizedBox()
                                  ],
                                ),
                              ),
                      ],
                    ),
                  ),
                ),
              )
            ],
          ),
        ),
      ),
    );
  }

  Future<void> Notification(String image,String name,String reason)async {
    String user =await FirebaseAuth.instance.currentUser!.email.toString();

    QuerySnapshot<Map<String, dynamic>> snapshot = await FirebaseFirestore.instance
        .collection('Usertokens')
        .get();
    String title = name +" wrongly classified";
    snapshot.docs.forEach((doc) {
      Map<String, dynamic> data = doc.data();
      if (data.containsKey('tokens') ) {
        String token = data['tokens'];
        print(token);

        sendmeassage(token,title,image,"Give your suggestions immediately");

      }
    });
  }
  Future<String?> getUserType() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    return prefs.getString('userType');
  }

}
