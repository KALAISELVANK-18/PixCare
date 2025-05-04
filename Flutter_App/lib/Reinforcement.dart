import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';

class Reinforcement extends StatefulWidget {
  final String imagePath;
  final List<DocumentSnapshot<Object?>> result;
  const Reinforcement({super.key,required this.imagePath,required this.result});
  @override
  State<Reinforcement> createState() => _ReinforcementState();
}

class _ReinforcementState extends State<Reinforcement> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        body: Container(

        ),
    );
  }
}
